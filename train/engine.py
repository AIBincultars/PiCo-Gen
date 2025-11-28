import os
import glob
import time
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2Config, get_cosine_schedule_with_warmup

from models.teacher import NotaGenLMHeadModel
from models.student import PiCoGen
from models.configuration import PiCoGenConfig

from data.dataset import SymphonyDataset, collate_fn
from .baseline import StandardTrainer
from .distiller import PiCoDistiller
from metrics.evaluator import Evaluator
import train.config as cfg


class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 准备保存路径 [核心修改]
        # 结构: checkpoints/{exp_name}/
        self.ckpt_dir = os.path.join("checkpoints", self.args.exp_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        print(f">>> Checkpoints will be saved to: {self.ckpt_dir}")

        # 2. 准备日志文件
        # 将日志也放在该目录下，方便管理
        self.log_path = os.path.join(self.ckpt_dir, "training_log.txt")

        print(f">>> Logs will be saved to: {self.log_path}")

        # 如果是第一次运行（非追加），写入表头
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("Epoch,Train_Loss,Val_Loss,PPL,Phy_Dist,Time\n")

        self.start_epoch = 1
        self.best_val_loss = float('inf')  # 初始化最佳 Loss 为无穷大

        # 3. 初始化 WandB
        if cfg.WANDB_LOGGING:
            wandb.init(
                project="picogen-baseline",
                name=self.args.exp_name,
                config={
                    "lr": self.args.lr,
                    "batch_size": self.args.batch_size,
                    "backbone": self.args.backbone,
                    "model_config": cfg.NAME
                }
            )

    def _get_model_name(self):
        """生成模型基础名称 (NotaGen 风格)"""
        # e.g. weights_picogen_baseline_v1_llama_lr1e-05_bz4
        name = (f"weights_picogen_{self.args.exp_name}_"
                f"{self.args.backbone}_"
                f"lr{self.args.lr}_"
                f"bz{self.args.batch_size}")
        return name

    def _load_teacher(self):
        print(">>> Loading Teacher (NotaGen)...")
        t_patch_cfg = GPT2Config(n_embd=1280, n_layer=20, n_head=20, n_positions=1024, vocab_size=1)
        t_char_cfg = GPT2Config(n_embd=1280, n_layer=6, n_head=20, vocab_size=128)
        teacher = NotaGenLMHeadModel(t_patch_cfg, t_char_cfg)

        if self.args.teacher_ckpt and os.path.exists(self.args.teacher_ckpt):
            print(f"Loading teacher weights from {self.args.teacher_ckpt}")
            ckpt = torch.load(self.args.teacher_ckpt, map_location='cpu')
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            teacher.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: Teacher checkpoints not loaded! Using random initialization.")

        teacher.to(self.device).eval()
        for p in teacher.parameters(): p.requires_grad = False
        return teacher

    def _init_student(self):
        print(f">>> Initializing Student (Backbone: {self.args.backbone})...")
        s_config = PiCoGenConfig(
            backbone_type=self.args.backbone,
            hidden_size=cfg.HIDDEN_SIZE // 2,
            num_hidden_layers=12,
            teacher_hidden_size=1280,
            patch_size=cfg.PATCH_SIZE,
            vocab_size=128
        )
        student = PiCoGen(s_config).to(self.device)
        return student

    def _resume_checkpoint(self, model, optimizer, scheduler):
        """
        断点续训逻辑：
        直接寻找 {ckpt_dir}/{model_name}_last.pth
        """
        model_name_prefix = self._get_model_name()
        last_ckpt_path = os.path.join(self.ckpt_dir, f"{model_name_prefix}_last.pth")

        if os.path.exists(last_ckpt_path):
            print(f">>> Found last checkpoint: {last_ckpt_path}")
            try:
                checkpoint = torch.load(last_ckpt_path, map_location=self.device)

                # 恢复模型和优化器状态
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint and optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                # 恢复训练进度
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1

                # 恢复最佳 Loss，防止误覆盖
                if 'best_val_loss' in checkpoint:
                    self.best_val_loss = checkpoint['best_val_loss']

                print(
                    f">>> Resumed training from Epoch {self.start_epoch}. Best Val Loss so far: {self.best_val_loss:.4f}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
        else:
            print(">>> No checkpoint found. Starting from scratch.")

    def run(self):
        # --- 数据加载 ---
        print(f"Loading dataset from {self.args.train_data}")
        train_ds = SymphonyDataset(self.args.train_data, patch_len=cfg.PATCH_LENGTH)

        eval_ds = None
        if hasattr(cfg, 'DATA_EVAL_INDEX_PATH') and os.path.exists(cfg.DATA_EVAL_INDEX_PATH):
            print(f"Loading eval dataset from {cfg.DATA_EVAL_INDEX_PATH}")
            eval_ds = SymphonyDataset(cfg.DATA_EVAL_INDEX_PATH, patch_len=cfg.PATCH_LENGTH)
        else:
            print("Warning: No eval dataset found. Metrics will be zero.")

        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=4)
        eval_loader = DataLoader(eval_ds, batch_size=self.args.batch_size, collate_fn=collate_fn) if eval_ds else None

        # --- 模型准备 ---
        student = self._init_student()
        optimizer = AdamW(student.parameters(), lr=self.args.lr)
        total_steps = self.args.epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

        # 恢复训练
        self._resume_checkpoint(student, optimizer, scheduler)

        # 评估器
        evaluator = Evaluator(student, eval_loader, self.device) if eval_loader else None

        # --- 选择 Trainer ---
        if self.args.mode == "distill":
            teacher = self._load_teacher()
            trainer = PiCoDistiller(
                student=student, teacher=teacher, dataloader=train_loader,
                optimizer=optimizer, device=self.device,
                alpha_kd=self.args.alpha_kd, beta_phy=self.args.beta_phy
            )
        else:
            trainer = StandardTrainer(student, train_loader, optimizer, self.device)

        # --- 训练循环 ---
        print(f">>> Start Training from Epoch {self.start_epoch} to {self.args.epochs}...")

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            start_time = time.time()

            # 1. 训练
            train_loss = trainer.train_epoch(epoch)
            scheduler.step()

            # 2. 评估
            val_loss = float('inf')
            ppl = 0.0
            phy_dist = 0.0

            if evaluator:
                metrics = evaluator.evaluate()
                val_loss = metrics['val_loss']
                ppl = metrics['ppl']
                phy_dist = metrics.get('phy_dist', 0.0)

            epoch_time = time.time() - start_time

            # 3. 日志
            log_str = (f"Epoch {epoch} | Time: {epoch_time:.1f}s | "
                       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                       f"PPL: {ppl:.2f}")
            print(log_str)

            with open(self.log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.5f},{val_loss:.5f},{ppl:.5f},{phy_dist:.5f},{epoch_time:.1f}\n")

            if cfg.WANDB_LOGGING:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "ppl": ppl,
                    "phy_dist": phy_dist,
                    "lr": optimizer.param_groups[0]['lr']
                })

            # 4. 保存模型 [核心修改]
            model_name = self._get_model_name()

            # 准备 Checkpoint 字典
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'best_val_loss': self.best_val_loss
            }

            # A. 始终保存/覆盖 last.pth (用于续训)
            last_path = os.path.join(self.ckpt_dir, f"{model_name}_last.pth")
            torch.save(ckpt_dict, last_path)

            # B. 如果性能更好，保存/覆盖 best.pth (用于推理)
            if val_loss < self.best_val_loss:
                print(f"Validation Loss Improved ({self.best_val_loss:.4f} -> {val_loss:.4f}). Saving best model...")
                self.best_val_loss = val_loss
                ckpt_dict['best_val_loss'] = val_loss  # 更新字典中的最佳值

                best_path = os.path.join(self.ckpt_dir, f"{model_name}_best.pth")
                torch.save(ckpt_dict, best_path)
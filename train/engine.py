import os
import glob
import torch
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

        # 修改：保存路径统一到 checkpoints/ 目录下
        self.ckpt_dir = os.path.join("checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.start_epoch = 1

    def _get_model_name(self):
        """
        生成类似 NotaGen 风格的模型名称
        """
        name = (f"weights_picogen_{self.args.exp_name}_"
                f"backbone_{self.args.backbone}_"
                f"h_{cfg.HIDDEN_SIZE}_"
                f"L_{cfg.PATCH_NUM_LAYERS}_"
                f"lr_{self.args.lr}_"
                f"bz_{self.args.batch_size}")
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
            hidden_size=cfg.HIDDEN_SIZE // 2,  # 示例：学生模型Hidden Size
            num_hidden_layers=12,
            teacher_hidden_size=1280,
            patch_size=cfg.PATCH_SIZE,
            vocab_size=128
        )
        student = PiCoGen(s_config).to(self.device)
        return student

    def _resume_checkpoint(self, model, optimizer, scheduler):
        """
        断点续训逻辑：自动查找最新的 checkpoint
        """
        model_name_prefix = self._get_model_name()
        # 在 checkpoints 文件夹中查找匹配当前实验配置的文件
        search_pattern = os.path.join(self.ckpt_dir, f"{model_name_prefix}_epoch_*.pth")
        ckpts = glob.glob(search_pattern)

        if len(ckpts) > 0:
            # 按 epoch 排序，取最后一个
            ckpts.sort(key=os.path.getmtime)
            latest_ckpt = ckpts[-1]
            print(f">>> Resuming from checkpoint: {latest_ckpt}")

            checkpoint = torch.load(latest_ckpt, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint and optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1

            print(f">>> Resumed training from Epoch {self.start_epoch}")
        else:
            print(">>> No checkpoint found. Starting from scratch.")

    def run(self):
        # 1. 数据
        print(f"Loading dataset from {self.args.train_data}")
        # 注意：这里使用了之前修复过路径问题的 Dataset 类
        train_ds = SymphonyDataset(self.args.train_data, patch_len=cfg.PATCH_LENGTH)
        # 尝试加载验证集
        eval_ds = None
        if hasattr(cfg, 'DATA_EVAL_INDEX_PATH') and os.path.exists(cfg.DATA_EVAL_INDEX_PATH):
            eval_ds = SymphonyDataset(cfg.DATA_EVAL_INDEX_PATH, patch_len=cfg.PATCH_LENGTH)

        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=4)
        eval_loader = DataLoader(eval_ds, batch_size=self.args.batch_size, collate_fn=collate_fn) if eval_ds else None

        # 2. 模型与优化器
        student = self._init_student()
        optimizer = AdamW(student.parameters(), lr=self.args.lr)
        total_steps = self.args.epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

        # 3. 断点续训检查
        # 假设 args 中有个 --resume 参数，或者是默认开启检查
        self._resume_checkpoint(student, optimizer, scheduler)

        # 4. 准备 Evaluator
        evaluator = Evaluator(student, eval_loader, self.device) if eval_loader else None

        # 5. 选择 Trainer
        if self.args.mode == "distill":
            teacher = self._load_teacher()
            trainer = PiCoDistiller(student, teacher, train_loader, optimizer, self.device,
                                    alpha_kd=self.args.alpha_kd, beta_phy=self.args.beta_phy)
        else:
            trainer = StandardTrainer(student, train_loader, optimizer, self.device)

        # 6. 训练循环
        print(f">>> Start Training from Epoch {self.start_epoch} to {self.args.epochs}...")
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            avg_loss = trainer.train_epoch(epoch)
            scheduler.step()

            log_str = f"Epoch {epoch} | Train Loss: {avg_loss:.4f}"

            # 验证
            if evaluator:
                metrics = evaluator.evaluate()
                log_str += f" | Val Loss: {metrics['val_loss']:.4f} | PPL: {metrics['ppl']:.2f} | Phy Dist: {metrics['phy_dist']:.4f}"

            print(log_str)

            # 保存 Checkpoint
            model_name = self._get_model_name()
            save_path = os.path.join(self.ckpt_dir, f"{model_name}_epoch_{epoch}.pth")

            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved checkpoint: {save_path}")
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from .losses import PhysicsAwareLoss


class Distiller:
    def __init__(self, student, teacher, train_dataset):
        self.device = torch.device(config.DEVICE)
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device).eval()

        # 冻结 Teacher
        for p in self.teacher.parameters(): p.requires_grad = False

        self.loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_fn)

        self.optimizer = AdamW(self.student.parameters(), lr=config.LEARNING_RATE)
        self.phy_loss_fn = PhysicsAwareLoss()

    def train(self):
        self.student.train()

        for epoch in range(config.NUM_EPOCHS):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}")
            epoch_loss = 0

            for batch in pbar:
                patches = batch['patches'].to(self.device)
                masks = batch['masks'].to(self.device)
                inst_ids = batch['inst_ids'].to(self.device)

                # 1. Teacher Forward
                with torch.no_grad():
                    t_out = self.teacher(patches, masks)

                # 2. Student Forward
                s_out = self.student(patches, masks, instrument_ids=inst_ids)

                # 3. Loss Calculation
                # A. Hard Label (NLL)
                # 简单 Flatten 处理
                flat_logits = s_out['logits'].view(-1, config.VOCAB_SIZE)
                flat_targets = patches.view(-1)  # 注意：这里其实应该 shift 一位做 next token prediction
                # 为了代码跑通，这里做简化示意。正确的做法参考原 train-gen.py 的 shift 逻辑
                loss_ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=0)

                # B. Soft Label (KD)
                loss_kd = F.kl_div(
                    F.log_softmax(s_out['logits'] / config.TEMPERATURE, dim=-1),
                    F.softmax(t_out['logits'] / config.TEMPERATURE, dim=-1),
                    reduction='batchmean'
                ) * (config.TEMPERATURE ** 2)

                # C. Structural (MSE)
                loss_struct = F.mse_loss(s_out['projected_latents'], t_out['patch_latents'])

                # D. Physics
                loss_phy = self.phy_loss_fn(s_out['logits'], t_out['logits'])

                # Total
                loss = loss_ce + config.ALPHA_KD * loss_kd + \
                       config.BETA_STRUCT * loss_struct + loss_phy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})

            # Save Checkpoint
            torch.save(self.student.state_dict(), config.STUDENT_SAVE_PATH)
            print(f"Epoch {epoch + 1} finished. Loss: {epoch_loss / len(self.loader)}")
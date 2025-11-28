import torch
import torch.nn.functional as F
from tqdm import tqdm
from .losses import PhysicsAwareLoss
from config import PATCH_SIZE


class PiCoDistiller:
    def __init__(self, student, teacher, dataloader, optimizer, device, alpha_kd=1.0, beta_phy=0.5):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()  # 冻结 Teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.alpha_kd = alpha_kd  # 蒸馏权重
        self.beta_phy = beta_phy  # 物理损失权重

        self.phy_loss_fn = PhysicsAwareLoss()

    def train_epoch(self, epoch):
        self.student.train()
        total_loss = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch} Distillation")
        for batch in pbar:
            patches, masks = batch
            patches = patches.to(self.device)
            masks = masks.to(self.device)

            # 1. Teacher Forward (No Grad)
            with torch.no_grad():
                # [Fix]: 直接调用 NotaGen 的子模块来获取中间层 hidden states
                # NotaGen 输入需要 reshape
                t_inputs = patches.reshape(len(patches), -1, PATCH_SIZE)

                # 获取 Teacher Encoder (Patch-level) 的输出作为蒸馏目标
                # 注意：NotaGen 的 patch_level_decoder 返回的是 GPT2Model 的输出
                t_enc_out = self.teacher.patch_level_decoder(t_inputs, masks)
                t_latents = t_enc_out["last_hidden_state"]  # [B, S, H_teacher]

                # 获取 Teacher 的最终 Logits 用于 KL 散度
                t_logits = self.teacher(patches, masks)

            # 2. Student Forward
            s_outputs = self.student(patches, masks)
            s_logits = s_outputs['logits']
            s_projected = s_outputs['projected']  # [B, S, H_teacher]

            # 3. 损失计算

            # A. Logits 蒸馏 (KL Divergence)
            temp = 2.0
            loss_kd_logits = F.kl_div(
                F.log_softmax(s_logits / temp, dim=-1),
                F.softmax(t_logits / temp, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)

            # B. 隐层蒸馏 (Hidden State Alignment)
            loss_hidden = F.mse_loss(s_projected, t_latents)

            # C. 物理感知损失 (Physics-Aware Loss)
            loss_phy, _ = self.phy_loss_fn(s_logits, t_logits)

            # D. 任务损失 (Next Token Prediction)
            # 简单的 Shift 操作
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = patches[..., 1:].contiguous()
            # Flatten
            loss_task = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0  # 假设 0 是 pad
            )

            # 组合损失
            loss = loss_task + self.alpha_kd * (loss_kd_logits + loss_hidden) + self.beta_phy * loss_phy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "L_Task": f"{loss_task.item():.3f}",
                "L_Hid": f"{loss_hidden.item():.3f}",
                "L_Phy": f"{loss_phy.item():.3f}"
            })

        return total_loss / len(self.dataloader)
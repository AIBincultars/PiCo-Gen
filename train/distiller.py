import torch
import torch.nn.functional as F
from tqdm import tqdm
from .losses import PhysicsAwareLoss
from config import PATCH_SIZE


class PiCoDistiller:
    def __init__(self, student, teacher, dataloader, optimizer, device, alpha_kd=1.0, beta_phy=0.5):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()

        # 确保 Teacher 不更新梯度
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.alpha_kd = alpha_kd
        self.beta_phy = beta_phy
        self.phy_loss_fn = PhysicsAwareLoss()

    def _get_teacher_components(self, teacher_model):
        """
        辅助函数：处理 PEFT/LoRA 包装，获取 Teacher 的核心组件。
        """
        # 如果是 PeftModel，通常模型在 teacher_model.base_model.model 中
        # 或者直接尝试访问属性
        if hasattr(teacher_model, "patch_level_decoder"):
            return teacher_model.patch_level_decoder, teacher_model

        # 尝试解包 PEFT
        if hasattr(teacher_model, "base_model") and hasattr(teacher_model.base_model, "model"):
            inner_model = teacher_model.base_model.model
            if hasattr(inner_model, "patch_level_decoder"):
                return inner_model.patch_level_decoder, teacher_model

        raise AttributeError("Cannot find 'patch_level_decoder' in Teacher model. Please check PEFT/LoRA wrapping.")

    def train_epoch(self, epoch):
        self.student.train()
        total_loss = 0

        # 获取 Teacher 的 Patch Encoder (用于隐层蒸馏)
        t_patch_decoder, _ = self._get_teacher_components(self.teacher)

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch} Distillation")
        for batch in pbar:
            patches, masks = batch
            patches = patches.to(self.device)
            masks = masks.to(self.device)

            # 1. Teacher Forward (No Grad)
            with torch.no_grad():
                t_inputs = patches.reshape(len(patches), -1, PATCH_SIZE)

                # 获取 Teacher 中间层特征 (Target for Hidden Loss)
                t_enc_out = t_patch_decoder(t_inputs, masks)
                t_latents = t_enc_out["last_hidden_state"]

                # 获取 Teacher 最终 Logits (Target for KL Loss)
                # 注意：如果 self.teacher 是 PEFT 模型，直接调用 forward 应该没问题
                t_logits = self.teacher(patches, masks)

            # 2. Student Forward
            s_outputs = self.student(patches, masks)
            s_logits = s_outputs['logits']
            s_projected = s_outputs['projected']

            # 3. 损失计算
            temp = 2.0
            loss_kd_logits = F.kl_div(
                F.log_softmax(s_logits / temp, dim=-1),
                F.softmax(t_logits / temp, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)

            loss_hidden = F.mse_loss(s_projected, t_latents)
            loss_phy, _ = self.phy_loss_fn(s_logits, t_logits)

            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = patches[..., 1:].contiguous()
            loss_task = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0
            )

            loss = loss_task + self.alpha_kd * (loss_kd_logits + loss_hidden) + self.beta_phy * loss_phy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "L_Task": f"{loss_task.item():.3f}",
                "L_Distill": f"{(loss_kd_logits + loss_hidden).item():.3f}",
                "L_Phy": f"{loss_phy.item():.3f}"
            })

        return total_loss / len(self.dataloader)
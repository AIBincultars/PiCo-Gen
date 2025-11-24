import torch
import torch.nn.functional as F
from tqdm import tqdm
from .losses import PhysicsAwareLoss


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
            # 假设 batch 是 (patches, masks)
            patches = batch[0].to(self.device)
            masks = batch[1].to(self.device)

            # 1. Teacher Forward (No Grad)
            with torch.no_grad():
                # NotaGen teacher 输出
                # 注意：需要确保 Teacher 的 forward 返回 patch_level 的 hidden_state
                t_outputs = self.teacher(patches, masks)
                t_logits = t_outputs.logits
                t_latents = t_outputs.patch_level_decoder_output  # 需修改 Teacher 代码使其返回此项

            # 2. Student Forward
            s_outputs = self.student(patches, masks)
            s_logits = s_outputs['logits']
            s_projected = s_outputs['projected_latents']

            # 3. 损失计算

            # A. Logits 蒸馏 (KL Divergence)
            # 软化 Teacher 分布
            temp = 2.0
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / temp, dim=-1),
                F.softmax(t_logits / temp, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)

            # B. 隐层蒸馏 (Hidden State Alignment)
            # 让 Student 的大脑模仿 Teacher 的大脑结构
            loss_hidden = F.mse_loss(s_projected, t_latents)

            # C. 物理感知损失 (Physics-Aware Loss)
            # 确保生成的统计规律符合幂律
            loss_phy, phy_stats = self.phy_loss_fn(s_logits, t_logits)

            # D. 任务损失 (Hard Label)
            # 这里简化为 Next Token Prediction，实际需做 Shift
            loss_task = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), patches.view(-1))

            # 组合损失
            loss = loss_task + self.alpha_kd * (loss_kd + loss_hidden) + self.beta_phy * loss_phy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": loss.item(),
                "KD": loss_kd.item(),
                "Phy": loss_phy.item()
            })

        return total_loss / len(self.dataloader)
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class PhysicsAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_phy = config.GAMMA_PHYSICS
        self.scale_ent = config.LAMBDA_ENTROPY

    def forward(self, logits, teacher_logits=None):
        probs = F.softmax(logits, dim=-1)
        vocab_size = logits.size(-1)
        pitch_vals = torch.arange(vocab_size, device=logits.device).float()

        # 1. 计算平滑度 (Smoothness Constraint)
        expected_pitch = torch.sum(probs * pitch_vals, dim=-1)
        intervals = torch.abs(expected_pitch[:, 1:] - expected_pitch[:, :-1])
        current_mean = intervals.mean()

        if teacher_logits is not None:
            with torch.no_grad():
                t_probs = F.softmax(teacher_logits, dim=-1)
                t_pitch = torch.sum(t_probs * pitch_vals, dim=-1)
                t_int = torch.abs(t_pitch[:, 1:] - t_pitch[:, :-1])
                target_mean = t_int.mean()
        else:
            target_mean = torch.tensor(2.5, device=logits.device)  # 经验值

        loss_smooth = F.mse_loss(current_mean, target_mean)

        # 2. 计算熵 (Entropy Maximization Objective)
        # H = -sum(p * log p)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

        # Loss = Smooth_Error - lambda * Entropy (Max Entropy -> Min -Entropy)
        return self.scale_phy * loss_smooth - self.scale_ent * entropy
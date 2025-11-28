import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsAwareLoss(nn.Module):
    """
    Based on 'Goal Orientation in Music Composition...'.
    Enforces Power Law distribution via Smoothness Constraint and Entropy Maximization.
    """

    def __init__(self, lambda_smooth=0.5, lambda_entropy=0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_entropy = lambda_entropy

    def forward(self, logits, teacher_logits=None):
        # logits: [B, S, V]
        probs = F.softmax(logits, dim=-1) + 1e-9
        vocab_size = logits.size(-1)

        # Differentiable Expected Pitch
        pitch_vals = torch.arange(vocab_size, device=logits.device).float()
        expected_pitch = torch.sum(probs * pitch_vals, dim=-1)  # [B, S]

        # Melodic Intervals
        intervals = torch.abs(expected_pitch[:, 1:] - expected_pitch[:, :-1])
        intervals = torch.clamp(intervals, min=1.0)

        # 1. Smoothness Constraint
        log_intervals = torch.log(intervals)
        smoothness_stat = log_intervals.mean()

        if teacher_logits is not None:
            with torch.no_grad():
                t_probs = F.softmax(teacher_logits, dim=-1)
                t_pitch = torch.sum(t_probs * pitch_vals, dim=-1)
                t_intervals = torch.abs(t_pitch[:, 1:] - t_pitch[:, :-1]).clamp(min=1.0)
                target_smoothness = torch.log(t_intervals).mean()
            loss_smooth = F.mse_loss(smoothness_stat, target_smoothness)
        else:
            # Empirical constant s0 approx 2.5
            loss_smooth = torch.abs(smoothness_stat - 2.5)

        # 2. Entropy Maximization
        entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
        loss_entropy = -entropy

        total_loss = self.lambda_smooth * loss_smooth + self.lambda_entropy * loss_entropy
        return total_loss, {"L_phy_sm": loss_smooth.item(), "L_phy_ent": loss_entropy.item()}
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from .physics import PhysicsMetrics


class Evaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.phy_metrics = PhysicsMetrics()

    def evaluate(self):
        """
        在验证集上进行评估
        :return: dict 包含 val_loss, ppl, phy_score
        """
        self.model.eval()
        total_loss = 0.0
        total_phy_score = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                patches, masks = batch
                patches = patches.to(self.device)
                masks = masks.to(self.device)

                # 前向传播
                outputs = self.model(patches, masks)
                logits = outputs['logits']  # [B, S, V]

                # 1. 计算 Cross Entropy Loss
                # Shift for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = patches[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0
                )

                # 2. 计算物理一致性指标 (基于 Logits)
                phy_stat = self.phy_metrics.calculate_pitch_interval_stat(logits)
                # 假设目标常数约为 2.5 (参考相关文献)
                phy_score = abs(phy_stat - 2.5)

                total_loss += loss.item()
                total_phy_score += phy_score
                total_steps += 1

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_phy = total_phy_score / total_steps if total_steps > 0 else 0.0

        # 计算 Perplexity
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float('inf')

        return {
            "val_loss": avg_loss,
            "ppl": ppl,
            "phy_dist": avg_phy  # 越小越好
        }
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from .physics import PhysicsMetrics
from .musicality import MusicalityMetrics


class Evaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        # 初始化各个指标计算器
        self.phy_metrics = PhysicsMetrics()
        self.music_metrics = MusicalityMetrics()

    def evaluate(self):
        """
        在验证集上进行全面评估
        :return: dict 包含 val_loss, ppl, phy_dist, diversity, entropy
        """
        self.model.eval()

        # 累加器
        metrics_sum = {
            "loss": 0.0,
            "phy_dist": 0.0,
            "diversity": 0.0,
            "entropy": 0.0
        }
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                patches, masks = batch
                patches = patches.to(self.device)
                masks = masks.to(self.device)

                # 1. 前向传播
                outputs = self.model(patches, masks)
                logits = outputs['logits']  # 形状: [Batch, Seq, Patch, Vocab]

                # 2. 计算基础 Loss (需要 Patch 级别的 Shift)
                # 移除最后一个时间步的 Logits 和第一个时间步的 Labels
                shift_logits = logits[:, :-1, :, :].contiguous()
                shift_labels = patches[:, 1:, :].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0
                )

                # --- [关键修改] 展平 Patch 维度用于指标计算 ---
                # 我们希望把数据看作连续的音符流，而不是分块的 Patch
                # 形状变换: [Batch, Seq, Patch, Vocab] -> [Batch, Seq*Patch, Vocab]
                b, s, p, v = logits.shape
                flat_logits = logits.view(b, s * p, v)

                # 获取预测序列 (Hard Prediction)
                # 形状: [Batch, Seq*Patch] -> 这样每个元素就是一个 Token ID
                pred_tokens = torch.argmax(flat_logits, dim=-1)

                # 3. 计算各项指标
                # 物理规律 (传入展平后的 Logits，以计算相邻 Token 间的音程)
                phy_stat = self.phy_metrics.calculate_pitch_interval_stat(flat_logits)
                phy_score = abs(phy_stat - 2.5)  # 目标值 2.5

                # 音乐性 & 生成质量 (传入展平后的 Token 序列)
                div_score = self.music_metrics.calculate_diversity(pred_tokens, n=3)
                ent_score = self.music_metrics.calculate_pitch_entropy(flat_logits)

                # 累加
                metrics_sum["loss"] += loss.item()
                metrics_sum["phy_dist"] += phy_score
                metrics_sum["diversity"] += div_score
                metrics_sum["entropy"] += ent_score
                total_steps += 1

        # 计算平均值
        avg_metrics = {k: v / total_steps for k, v in metrics_sum.items() if total_steps > 0}

        # 计算 PPL
        try:
            ppl = math.exp(avg_metrics.get("loss", 0))
        except OverflowError:
            ppl = float('inf')

        return {
            "val_loss": avg_metrics.get("loss", 0),
            "ppl": ppl,
            "phy_dist": avg_metrics.get("phy_dist", 0),
            "diversity": avg_metrics.get("diversity", 0),
            "entropy": avg_metrics.get("entropy", 0)
        }
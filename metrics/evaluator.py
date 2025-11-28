import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from .physics import PhysicsMetrics
from .musicality import MusicalityMetrics  # [新增导入]


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
                logits = outputs['logits']  # [B, S, V]

                # 2. 计算基础 Loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = patches[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0
                )

                # 3. 获取预测序列 (Hard Prediction) 用于计算多样性
                # 使用 argmax 获取概率最大的 token
                pred_tokens = torch.argmax(logits, dim=-1)

                # 4. 计算各项指标
                # 物理规律 (Power Law 距离)
                phy_stat = self.phy_metrics.calculate_pitch_interval_stat(logits)
                phy_score = abs(phy_stat - 2.5)  # 目标值 2.5

                # 音乐性 & 生成质量
                div_score = self.music_metrics.calculate_diversity(pred_tokens, n=3)
                ent_score = self.music_metrics.calculate_pitch_entropy(logits)

                # 累加
                metrics_sum["loss"] += loss.item()
                metrics_sum["phy_dist"] += phy_score
                metrics_sum["diversity"] += div_score
                metrics_sum["entropy"] += ent_score
                total_steps += 1

        # 计算平均值
        avg_metrics = {k: v / total_steps for k, v in metrics_sum.items()}

        # 计算 PPL
        try:
            ppl = math.exp(avg_metrics["loss"])
        except OverflowError:
            ppl = float('inf')

        return {
            "val_loss": avg_metrics["loss"],
            "ppl": ppl,
            "phy_dist": avg_metrics["phy_dist"],  # 越接近 0 越符合物理规律
            "diversity": avg_metrics["diversity"],  # 越高越好 (0-1)
            "entropy": avg_metrics["entropy"]  # 适中为好 (过低单调，过高杂乱)
        }
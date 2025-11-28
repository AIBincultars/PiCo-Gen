import torch
import numpy as np


class PhysicsMetrics:
    """
    计算音乐生成的物理统计指标，主要关注音高变化的幂律分布特性。
    """

    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size

    def calculate_pitch_interval_stat(self, logits):
        """
        计算音程变化的统计量。
        :param logits: [Batch, Seq_Len, Vocab_Size]
        :return: smoothness_score (越低越好，越接近 2.5 表示越符合幂律)
        """
        # 使用 Softmax 获取概率分布
        probs = torch.softmax(logits, dim=-1)

        # 计算期望音高 (Expected Pitch)
        # 假设 vocab 0-127 对应 MIDI 音高，或者是相对的 Token ID
        pitch_vals = torch.arange(self.vocab_size, device=logits.device).float()
        expected_pitch = torch.sum(probs * pitch_vals, dim=-1)  # [Batch, Seq]

        # 计算相邻音符的音程 (Intervals)
        # diff: [Batch, Seq-1]
        intervals = torch.abs(expected_pitch[:, 1:] - expected_pitch[:, :-1])

        # 避免 log(0)
        intervals = torch.clamp(intervals, min=1e-6)

        # 计算 Log Intervals 的均值
        # 在 1/f 噪声理论中，音程大小的对数分布通常具有特定的统计特征
        log_intervals = torch.log(intervals)
        smoothness_stat = log_intervals.mean()

        return smoothness_stat.item()

    def compute_power_law_fit(self, sequences):
        """
        (可选) 对生成的具体序列计算 1/f 拟合度 R-squared
        :param sequences: [Batch, Seq_Len] (Token IDs)
        """
        # 这里的实现较复杂，通常用于离线评估。
        # 实时训练中主要使用 calculate_pitch_interval_stat
        pass
import torch
import torch.nn.functional as F


class MusicalityMetrics:
    """
    计算音乐性和生成质量相关的指标：多样性、熵、密度等。
    """

    def __init__(self):
        pass

    def calculate_diversity(self, sequences, n=3):
        """
        计算 Distinct-N 指标，衡量生成的多样性。
        :param sequences: [Batch, Seq_Len] Token IDs
        :param n: n-gram 的长度，默认为 3
        :return: distinct_score (0.0 - 1.0)，越高越好
        """
        batch_size = sequences.size(0)
        total_ngrams = 0
        unique_ngrams = set()

        for i in range(batch_size):
            # 将 Tensor 转换为列表，移除 Padding (假设 0 是 padding)
            tokens = [t.item() for t in sequences[i] if t.item() != 0]

            if len(tokens) < n:
                continue

            # 统计该样本中的 n-grams
            for j in range(len(tokens) - n + 1):
                ngram = tuple(tokens[j:j + n])
                unique_ngrams.add(ngram)
                total_ngrams += 1

        if total_ngrams == 0:
            return 0.0

        # 唯一 n-gram 数量 / 总 n-gram 数量
        return len(unique_ngrams) / total_ngrams

    def calculate_pitch_entropy(self, logits):
        """
        计算预测分布的平均信息熵 (Shannon Entropy)。
        衡量模型预测的不确定性和丰富度。
        :param logits: [Batch, Seq_Len, Vocab_Size]
        :return: entropy score
        """
        # 转换为概率分布
        probs = F.softmax(logits, dim=-1)  # [B, S, V]

        # 计算熵: H(p) = - sum(p * log(p))
        # 加 1e-9 防止 log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        # 计算整个 Batch 的平均熵 (忽略 padding 的掩码处理由外部控制，这里简化为求均值)
        return entropy.mean().item()

    def calculate_note_density(self, sequences):
        """
        计算音符密度 (非空 Token 的比例)。
        """
        # 假设 0 是 padding/special token
        non_pad = (sequences != 0).sum().float()
        total = sequences.numel()
        return (non_pad / total).item()
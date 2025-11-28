import torch
import torch.nn.functional as F
from tqdm import tqdm


class StandardTrainer:
    def __init__(self, student, dataloader, optimizer, device):
        """
        Baseline 训练器：不依赖 Teacher，仅使用 CrossEntropy 训练 Student。
        """
        self.student = student
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, epoch):
        self.student.train()
        total_loss = 0

        # 进度条显示当前 Epoch 状态
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch} [Baseline]")

        for patches, masks in pbar:
            # 1. 数据搬运到 GPU
            patches = patches.to(self.device)  # [Batch, Seq_Len]
            masks = masks.to(self.device)  # [Batch, Seq_Len]

            # 2. 前向传播 (Forward)
            outputs = self.student(patches, masks)
            logits = outputs['logits']  # [Batch, Seq_Len, Vocab_Size]

            # 3. 计算损失 (Loss)
            # 这里使用标准的分类损失：预测每个位置的 Token ID
            # view(-1, ...) 将 Batch 和 Sequence 维度展平，以便计算 CrossEntropy
            vocab_size = logits.size(-1)

            # 注意：ignore_index=0 是因为你的 tokenizer 中 pad_token_id 通常为 0
            # 如果你的 padding id 不是 0，请修改这里
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                patches.view(-1),
                ignore_index=0
            )

            # 4. 反向传播与优化 (Backward & Step)
            self.optimizer.zero_grad()
            loss.backward()

            # 可选：梯度裁剪，防止梯度爆炸 (特别是对 RNN/RWKV 类模型很重要)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 5. 更新统计
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 返回当前 Epoch 的平均 Loss
        return total_loss / len(self.dataloader)
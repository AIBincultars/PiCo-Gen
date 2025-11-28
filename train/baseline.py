import torch
import torch.nn.functional as F
from tqdm import tqdm


class StandardTrainer:
    def __init__(self, student, dataloader, optimizer, device):
        self.student = student
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, epoch):
        self.student.train()
        total_loss = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch} [Baseline]")

        for patches, masks in pbar:
            # patches: [B, S, P] (batch, seq_len, patch_size)
            patches = patches.to(self.device)
            masks = masks.to(self.device)

            outputs = self.student(patches, masks)
            logits = outputs['logits']  # [B, S, P, V]

            # --- 核心修正：Shift & Flatten ---
            # 目标：根据第 t 个 patch (及之前的) 预测第 t+1 个 patch
            # Input:  [x0, x1, x2, ...]
            # Target: [x1, x2, x3, ...]

            # Logits: 预测结果 [p1_pred, p2_pred, ...] (去掉最后一个时间步的无效预测)
            shift_logits = logits[:, :-1, :, :].contiguous()

            # Labels: 真实结果 [p1, p2, ...] (去掉第一个时间步作为输入)
            shift_labels = patches[:, 1:, :].contiguous()

            # Flatten to [N, Vocab_Size] and [N]
            # N = B * (S-1) * P
            vocab_size = shift_logits.size(-1)

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=0  # Padding ID
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return total_loss / len(self.dataloader)
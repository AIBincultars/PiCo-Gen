import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 引入项目模块
from config import *
from models.configuration import PiCoGenConfig
from models.student import PiCoGen
from data.dataset import InstructionDataset, collate_fn


def train_sft(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化随机的 Student (Backbone)
    # 不加载任何预训练权重，直接从头开始
    print("Initializing PiCoGen Student from scratch...")
    config = PiCoGenConfig(
        backbone_type="llama",  # 或 mamba 等
        vocab_size=128,  # ASCII 字符级
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 4,
        num_hidden_layers=PATCH_NUM_LAYERS,  # e.g. 12 or 20
        num_attention_heads=16,
        max_position_embeddings=PATCH_LENGTH
    )
    model = PiCoGen(config)
    model.to(device)
    model.train()

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # 2. 加载指令数据集
    print(f"Loading Instruction Dataset from {args.data_path}")
    dataset = InstructionDataset(args.data_path, patch_len=PATCH_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # 简单的学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. 训练循环
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [SFT]")
        total_loss = 0

        for patches, masks in pbar:
            patches = patches.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(patches, masks)
            logits = outputs['logits']  # [B, S, P, V]

            # Shift & Flatten for Loss
            # 预测下一个 Patch
            shift_logits = logits[:, :-1, :, :].contiguous()
            shift_labels = patches[:, 1:, :].contiguous()

            # Flatten
            vocab_size = shift_logits.size(-1)
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=0  # Pad Token ID
            )

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪防止梯度爆炸 (对于从头训练很重要)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.6f}"})

        scheduler.step()

        # 保存权重
        save_dir = "weights_sft"
        os.makedirs(save_dir, exist_ok=True)
        if (epoch + 1) % 1 == 0:  # 每 Epoch 保存
            save_path = os.path.join(save_dir, f"student_sft_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to instruction jsonl")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    train_sft(args)
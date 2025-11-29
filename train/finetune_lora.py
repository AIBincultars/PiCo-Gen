import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config
from peft import get_peft_model, LoraConfig, TaskType

from config import *  # 导入你的全局配置
from models.teacher import NotaGenLMHeadModel
from data.dataset import InstructionDataset, collate_fn

# LoRA 特定配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# NotaGen 的 Transformer 结构通常包含 c_attn (Linear layer in GPT2 attention)
TARGET_MODULES = ["c_attn"]


def train_lora(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载 Base Model (Teacher / NotaGen)
    print("Loading NotaGen Model for LoRA Tuning...")
    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, max_length=PATCH_LENGTH, n_embd=HIDDEN_SIZE,
                              vocab_size=1)
    byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, max_length=PATCH_SIZE + 1, hidden_size=HIDDEN_SIZE,
                             vocab_size=128)

    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)

    # 加载预训练权重
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        ckpt = torch.load(args.pretrained_path, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        print("Warning: Training from scratch (Check your PRETRAINED_PATH).")

    model.to(device)

    # 2. 配置并应用 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )

    # 将 LoRA 适配器应用到模型上
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 数据集加载
    print(f"Loading Instruction Data from {args.data_path}")
    dataset = InstructionDataset(args.data_path, patch_len=PATCH_LENGTH)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 5. 训练循环
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs} [LoRA]")
        total_loss = 0

        for patches, masks in pbar:
            patches = patches.to(device)
            masks = masks.to(device)

            # NotaGen Forward 计算
            # 注意: NotaGenLMHeadModel 的 forward 会在内部调用 char_level_decoder 并计算 Loss
            # 输入 patches 同时作为 Input 和 Target (Shifted inside model)
            outputs = model(patches, masks)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 保存权重
        save_dir = "weights_lora"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"lora_epoch_{epoch + 1}.pth")

        # 保存 PEFT 权重 (Adapter only)
        model.save_pretrained(os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
        print(f"Saved LoRA adapter to {os.path.join(save_dir, f'checkpoint_{epoch + 1}')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to instruction jsonl file")
    parser.add_argument('--pretrained_path', type=str, default="", help="Path to original NotaGen weights")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_lora(args)
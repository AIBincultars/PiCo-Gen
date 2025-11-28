import os
import torch
import time
import argparse
from transformers import GPT2Config
from models.student import PiCoGen
from models.configuration import PiCoGenConfig
from data.tokenizer import Patchilizer
import train.config as cfg  # 复用训练配置中的一些常量


def generate_music(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 解析模型路径以确定输出文件夹
    # 假设权重文件名类似：weights_picogen_expname_...pth
    weights_path = args.weights_path
    model_filename = os.path.basename(weights_path).replace('.pth', '')

    # 构建输出目录: output/<model_name>/
    output_dir = os.path.join("output", model_filename)
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Output directory: {output_dir}")

    # 2. 初始化模型
    print(">>> Loading Model...")
    # 注意：这里需要手动匹配训练时的配置，或者从checkpoint中加载config(如果保存了的话)
    # 为了简化，这里使用默认配置，请确保与训练时一致
    s_config = PiCoGenConfig(
        backbone_type=args.backbone,  # 必须与训练时一致
        hidden_size=cfg.HIDDEN_SIZE // 2,
        num_hidden_layers=12,
        teacher_hidden_size=1280,
        patch_size=cfg.PATCH_SIZE,
        vocab_size=128
    )

    model = PiCoGen(s_config).to(device)

    # 加载权重
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 兼容只保存了 state_dict 的情况

    model.eval()
    print(">>> Model Loaded Successfully.")

    # 3. 准备 Tokenizer 和 Prompt
    patchilizer = Patchilizer()

    # 示例 Prompt：可以根据需求修改，这里使用简单的开头
    # 实际应用中可能需要 "Period-Composer-Instrumentation" 格式
    prompts = [
        "X:1\nL:1/8\nQ:1/4=100\nK:C\n",  # 基础 C 大调
        # 可以添加更多 prompt
    ]

    # 4. 生成循环
    print(f">>> Generating {args.num_samples} samples per prompt...")

    for p_idx, prompt_text in enumerate(prompts):
        for i in range(args.num_samples):
            # 编码 Prompt
            # 注意：PiCoGen 是基于 Patch 的，这里需要将文本转换为 patch IDs
            # 这里的 encode_generate 需要你自己确保 tokenizer 里的实现是兼容的
            # 假设 patchilizer.encode_generate 返回 list of token ids
            input_patches = patchilizer.encode_generate(prompt_text)
            input_patches = torch.tensor([input_patches], dtype=torch.long).to(device)  # [1, Seq]

            # 生成 (需要模型类中实现 generate 方法，或者在这里手写 loop)
            # 由于之前提到的 GenericStudent/PiCoGen 没有自带 generate
            # 这里简单模拟一个自回归生成过程 (伪代码/简化版)

            # 实际上，你应该在 PiCoGen 类中实现一个 generate 函数
            # 这里为了演示保存逻辑，假设我们获得了一个生成的 token 序列
            # generated_patches = model.generate(input_patches, max_length=512)

            # [临时方案]: 如果模型还没有 generate 方法，这里无法真正生成。
            # 为了让脚本能跑通，我们只保存 prompt。
            # 请在 models/student.py 中实现 generate 方法 (参考 NotaGen 的实现)
            generated_text = prompt_text + "\n% Generated content placeholder"

            # 5. 保存结果
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{model_filename}_prompt{p_idx}_sample{i}_{timestamp}.abc"
            save_path = os.path.join(output_dir, filename)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default="llama", help="Model backbone type")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")

    args = parser.parse_args()
    generate_music(args)
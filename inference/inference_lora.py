import os
import time
import torch
import sys
import argparse
from peft import PeftModel

# --- 路径 Hack (确保能导入项目根目录模块) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Patchilizer
from config import *  # 导入全局配置 (PATCH_SIZE, HIDDEN_SIZE 等)
from models.teacher import NotaGenLMHeadModel
from transformers import GPT2Config

# --- 默认配置 ---
# 你可以在这里修改，或者通过命令行参数传入
DEFAULT_BASE_MODEL_PATH = "weights/notagen_pretrained.pth"  # 原始 NotaGen 权重
DEFAULT_LORA_PATH = "weights_lora/checkpoint_final"  # LoRA 训练保存的路径
OUTPUT_FOLDER = "outputs_lora"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
patchilizer = Patchilizer()


def load_merged_model(base_path, lora_path):
    print(f"[1/3] Loading Base NotaGen Model from {base_path}...")
    # 配置必须与训练时一致
    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, max_length=PATCH_LENGTH, n_embd=HIDDEN_SIZE,
                              vocab_size=1)
    byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, max_length=PATCH_SIZE + 1, hidden_size=HIDDEN_SIZE,
                             vocab_size=128)

    base_model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)

    # 加载基座权重
    if os.path.exists(base_path):
        ckpt = torch.load(base_path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        base_model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"Base model weights not found at {base_path}")

    # 临时移到 device 以便合并
    base_model.to(device)

    print(f"[2/3] Loading LoRA Adapter from {lora_path}...")
    try:
        # 使用 PEFT 加载 Adapter
        # 注意：NotaGen 不是标准的 HF 模型，但只要结构兼容，PEFT 通常能识别 target_modules
        model = PeftModel.from_pretrained(base_model, lora_path)

        print("[3/3] Merging LoRA weights into Base Model...")
        # 关键步骤：合并权重并卸载 PEFT 包装
        # 这样 model 变回 NotaGenLMHeadModel 实例，拥有自定义的 generate 方法
        model = model.merge_and_unload()

    except Exception as e:
        print(f"Error loading LoRA: {e}")
        print("Tip: Ensure 'adapter_config.json' and 'adapter_model.bin' exist in LORA_PATH.")
        exit(1)

    model.eval()
    return model


def text_to_patches(text):
    """
    将文本指令转换为 Patch Tensor，逻辑模仿 inference.py 的 patchilize_metadata
    """
    # 1. 转为 ASCII 码列表
    byte_list = [ord(c) for c in text]

    # 2. 分块 (Patch Size)
    patches = []
    for i in range(0, len(byte_list), PATCH_SIZE):
        chunk = byte_list[i: i + PATCH_SIZE]
        # Padding: 如果不足 Patch Size，用 special_token_id 填充
        if len(chunk) < PATCH_SIZE:
            chunk += [patchilizer.special_token_id] * (PATCH_SIZE - len(chunk))
        patches.append(chunk)

    return patches


def inference_lora(args):
    # 1. 加载模型
    model = load_merged_model(args.base_path, args.lora_path)

    # 2. 准备指令
    # 为了触发续写，我们在指令后加换行符，模拟训练数据的格式
    instruction = args.instruction.strip() + "\n"
    print(f"\nTarget Instruction:\n{'-' * 20}\n{instruction}{'-' * 20}\n")

    # 3. 构造 Input Patches
    # 开头加上 BOS Patch (NotaGen 惯例)
    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

    prompt_patches = text_to_patches(instruction)
    prompt_patches.insert(0, bos_patch)

    # 转为 Tensor [1, Seq_Len, Patch_Size] (Wait, inference.py uses flattened input for model.generate?)
    # 检查 models/teacher.py 的 generate 签名: generate(self, patches, ...)
    # 并且在 inference.py 中 input_patches 被 reshape 成了 (1, -1) ？？？
    # 不，model.generate 内部第一步是 patches.reshape(len(patches), -1, PATCH_SIZE)
    # 所以传入 (1, N * P) 或者 (1, N, P) 都可以，只要总元素数对就行。
    # 这里我们构造标准的 (1, Seq_Len * Patch_Size) 以匹配 inference.py 的习惯

    input_tensor = torch.tensor(prompt_patches, device=device).flatten().unsqueeze(0)  # [1, Total_Tokens]

    # 记录已生成的文本（包含指令）
    generated_text_buffer = list(instruction)

    print("Generating...")
    start_time = time.time()

    # 4. 生成循环
    # NotaGen 的 generate 是 Patch 级别的
    while True:
        # 调用模型的 generate (一次生成一个 Patch)
        # 注意：这里调用的是 models/teacher.py 中的 NotaGenLMHeadModel.generate
        try:
            predicted_patch = model.generate(
                input_tensor,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature
            )
        except Exception as e:
            print(f"Generation Error: {e}")
            break

        # 检查是否结束 (BOS + EOS 组合通常作为特殊的结束标志)
        if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
            print("\n[End of Generation]")
            break

        # 解码并打印
        next_text = patchilizer.decode([predicted_patch])

        # 实时打印到控制台
        print(next_text, end='', flush=True)
        generated_text_buffer.extend(next_text)

        # 拼接新生成的 Patch 到输入，用于下一次自回归
        new_patch_tensor = torch.tensor(predicted_patch, device=device).unsqueeze(0)  # [1, 16]
        input_tensor = torch.cat([input_tensor, new_patch_tensor], dim=1)

        # 长度保护
        if len(generated_text_buffer) > 100000:  # 约 100k 字符限制
            print("\n[Length Limit Reached]")
            break

    # 5. 保存结果
    time_str = time.strftime("%Y%m%d-%H%M%S")
    filename = f"lora_gen_{time_str}.abc"
    save_path = os.path.join(OUTPUT_FOLDER, filename)

    full_content = "".join(generated_text_buffer)

    # 可选：提取 ABC 部分 (去除指令)
    # 如果你只想要 ABC，可以 split 指令
    abc_only = full_content
    if instruction in full_content:
        abc_only = full_content.split(instruction, 1)[1]

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(abc_only)

    print(f"\n\nSaved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_MODEL_PATH,
                        help="Path to original NotaGen weights")
    parser.add_argument('--lora_path', type=str, default=DEFAULT_LORA_PATH, help="Path to LoRA checkpoint folder")
    parser.add_argument('--instruction', type=str, default="Generate a classical piano piece in C Major.",
                        help="The instruction prompt")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.9)

    args = parser.parse_args()

    inference_lora(args)

import os
import sys
import torch
import time
import argparse
import subprocess

# --- [关键修复] 将项目根目录加入 sys.path ---
# 获取当前脚本所在目录 (inference/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (PiCo-Gen/)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
# 将根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------------

from transformers import GPT2Config
from models.student import PiCoGen
from models.configuration import PiCoGenConfig
from data.tokenizer import Patchilizer
import train.config as cfg


def run_abc2xml(abc_path):
    """
    调用项目中的 data/abc2xml.py 将 abc 转换为 xml
    自动适配运行路径，无论是在 root 还是 inference 目录下运行都能找到
    """
    # 尝试在当前目录下找 data (Root运行模式)
    converter_script = os.path.join("data", "abc2xml.py")

    # 如果没找到，尝试在父目录下找 data (Inference目录运行模式)
    if not os.path.exists(converter_script):
        # 获取项目根目录下的 data 路径
        converter_script = os.path.join(project_root, "data", "abc2xml.py")

    if not os.path.exists(converter_script):
        print(f"[Warning] Converter script not found at {converter_script}. Skipping XML conversion.")
        return

    try:
        subprocess.run([sys.executable, converter_script, abc_path], check=True)
        print(f" -> Converted to XML: {abc_path.replace('.abc', '.xml')}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to convert {abc_path} to XML: {e}")


def generate_music(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备输出目录
    weights_path = args.weights_path
    model_filename = os.path.basename(weights_path).replace('.pth', '')

    # 输出目录也做路径适配
    # 如果在 inference 下运行，output 应该在 ../output，或者直接用绝对路径
    # 这里为了简单，我们统一保存到项目根目录下的 output
    output_root = os.path.join(project_root, "output")
    output_dir = os.path.join(output_root, model_filename)
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载模型
    print(f">>> Loading Model from {weights_path} ...")
    s_config = PiCoGenConfig(
        backbone_type=args.backbone,
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
        model.load_state_dict(checkpoint)

    model.eval()
    print(">>> Model Loaded Successfully.")

    # 3. 准备 Prompt
    patchilizer = Patchilizer()
    prompts = []

    if args.prompt:
        # 命令行输入的 Prompt
        raw_prompt = args.prompt.replace('\\n', '\n')
        prompts.append(raw_prompt)
        print(f">>> Using Custom Prompt from CLI:\n{raw_prompt}")
    elif args.prompt_file:
        # 从文件读取 Prompt
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompts.append(f.read())
            print(f">>> Using Prompt from file: {args.prompt_file}")
        else:
            print(f"[Error] Prompt file {args.prompt_file} not found.")
            return
    else:
        # 默认 Prompt
        default_prompt = "X:1\nL:1/8\nQ:1/4=100\nM:4/4\nK:C\n"
        prompts.append(default_prompt)
        print(f">>> Using Default Prompt:\n{default_prompt}")

    # 4. 生成循环
    print(f">>> Generating {args.num_samples} samples...")

    for p_idx, prompt_text in enumerate(prompts):
        for i in range(args.num_samples):
            start_time = time.time()

            # 编码
            input_patches_list = patchilizer.encode_generate(prompt_text)
            input_patches = torch.tensor([input_patches_list], dtype=torch.long).to(device)

            # 生成
            try:
                generated_patches_list = model.generate(
                    input_patches,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                generated_text = patchilizer.decode(generated_patches_list)

            except Exception as e:
                print(f"Error during generation: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 保存
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename_base = f"{model_filename}_p{p_idx}_s{i}_{timestamp}"
            abc_filename = f"{filename_base}.abc"
            save_path = os.path.join(output_dir, abc_filename)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            print(f"[{i + 1}/{args.num_samples}] Saved ABC: {save_path}")

            # 转换 XML
            run_abc2xml(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--backbone", type=str, default="llama", help="Model backbone type")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--max_length", type=int, default=512, help="Max length (patches)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")

    parser.add_argument("--prompt", type=str, default=None, help="Custom ABC prompt string")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to a file containing the prompt")

    args = parser.parse_args()
    generate_music(args)
import os
import sys
import torch
import time
import argparse
import subprocess
import re  # [关键修正] 引入正则模块，用于清洗生成的训练标记

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


# =============== ABC 后处理（结构化 + 强清洗） ===============

def split_prompt_metadata_and_seed(prompt_text: str):
    """
    按 tokenizer.encode_generate 的逻辑：metadata 是 [V: 或 [r: 之前的行，seed 是之后的正文行
    （你现在用的 tokenizer 也是这么 split 的）:contentReference[oaicite:1]{index=1}
    """
    lines = prompt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = list(filter(None, lines))

    tunebody_index = None
    for i, line in enumerate(lines):
        if line.startswith("[V:") or line.startswith("[r:"):
            tunebody_index = i
            break

    if tunebody_index is None:
        tunebody_index = len(lines)

    metadata_lines = lines[:tunebody_index]
    seed_lines = lines[tunebody_index:]

    metadata = "\n".join(metadata_lines).rstrip() + "\n"
    seed = ("\n".join(seed_lines).rstrip() + "\n") if seed_lines else ""
    return metadata, seed


def extract_generated_tunebody(generated_text: str) -> str:
    """
    只取模型输出里第一次出现 [V: 之后的正文，防止模型把 metadata 乱写进来。
    """
    # 先去掉训练辅助标记 [r:...]
    text = re.sub(r"\[r:[^\]]+\]", "", generated_text)

    pos = text.find("[V:")
    if pos < 0:
        return ""
    return text[pos:]


def sanitize_tunebody(body: str) -> str:
    """
    目标：尽量保留音符/节奏，只去掉会让 abc2xml 报错的“元信息碎片/乱码”。
    不做逐字符白名单（避免把 clef/name 剪成 cef/ae 这种伪音符残片）。
    """
    body = body.replace("\r\n", "\n").replace("\r", "\n")

    # 1) 去掉训练标记
    body = re.sub(r"\[r:[^\]]+\]", "", body)

    # 2) 去掉控制字符（保留 \n \t）
    body = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", body)

    # 3) 禁用装饰/引号（很容易触发 misplaced）
    body = body.replace("+", "")
    body = body.replace('"', "")

    # 4) 删除正文里混入的 V: 定义行（只允许 [V:n] 这种切换）
    body = re.sub(r"(?m)^\s*V:[^\n]*\n", "", body)

    # 5) 干掉“任意 key=value 元信息碎片”（关键！可杀掉 cef=ebe / name=V1 / clef=treble 等）
    #    注意：key 至少2个字母，避免误伤 ABC 的 "=C" 还原号（key 只有 1 个字符）
    body = re.sub(r"\b[A-Za-z]{2,}\s*=\s*[^\s\]]+", "", body)

    # 6) 修复你日志里这类垃圾片段
    body = re.sub(r"\[:\s*\d+\s*\]", "", body)                 # remove "[:1]"
    body = re.sub(r"(?m)(\[V:\d+\])\s*:\s*\d+", r"\1 ", body)  # "[V:1] :1" -> "[V:1] "

    # 7) 去掉 "[49" 这类：'[' 后直接跟数字/冒号肯定非法（但保留 [CEG] 这种和弦）
    body = re.sub(r"\[(?=\s*[\d:])", "", body)

    # 8) 清理多余空白
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)

    body = body.strip("\n") + "\n"
    return body



def compose_final_abc(prompt_text: str, generated_text: str) -> str:
    """
    最终 ABC = prompt metadata(锁死) + seed(锁死) + 模型续写正文(清洗后)
    """
    metadata, seed = split_prompt_metadata_and_seed(prompt_text)

    # 只取模型输出里第一次出现 [V: 之后的正文
    gen_body = extract_generated_tunebody(generated_text)
    gen_body = sanitize_tunebody(gen_body) if gen_body else ""

    # 不做“前缀去重裁剪”（那一步在某些情况下会误删正文），最多只去掉完全重复的 seed 行
    seed_line = seed.strip()
    if seed_line and gen_body.lstrip().startswith(seed_line):
        gen_body = gen_body.lstrip()[len(seed_line):].lstrip()
        if gen_body and not gen_body.endswith("\n"):
            gen_body += "\n"

    final_body = (seed if seed else "") + (gen_body if gen_body else "")
    if not final_body.strip():
        # 兜底：至少给一个休止小节
        final_body = "[V:1] z8 |\n"

    return metadata + "\n" + final_body



# =============== abc2xml 调用 ===============

def run_abc2xml(abc_path: str):
    converter_script = os.path.join(project_root, "data", "abc2xml.py")
    if not os.path.exists(converter_script):
        # 兼容从根目录运行
        converter_script = os.path.join("data", "abc2xml.py")

    if not os.path.exists(converter_script):
        print(f"[Warning] Converter script not found: {converter_script}. Skipping XML conversion.")
        return

    xml_path = abc_path.replace(".abc", ".xml")
    try:
        with open(xml_path, "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, converter_script, abc_path], stdout=f, check=True)
        print(f" -> Converted to XML: {xml_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to convert {abc_path} to XML: {e}")
    except Exception as e:
        print(f"[Error] Unexpected error during XML conversion: {e}")


# =============== 主生成逻辑 ===============

def generate_music(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出目录
    weights_path = args.weights_path
    model_filename = os.path.basename(weights_path).replace(".pth", "")
    output_root = os.path.join(project_root, "output")
    output_dir = os.path.join(output_root, model_filename)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
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

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(">>> Model Loaded Successfully.")

    # Prompt（你 CLI 传入什么就用什么）
    patchilizer = Patchilizer()
    if args.prompt:
        prompt_text = args.prompt.replace("\\n", "\n")
        print(f">>> Using Prompt:\n{prompt_text}")
    elif args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        print(f">>> Using Prompt from file: {args.prompt_file}")
    else:
        prompt_text = "X:1\nT:demo\nM:4/4\nL:1/8\nK:C\nV:1 clef=treble name=\"V1\"\n\n[V:1] C2 E2 G2 |\n"
        print(f">>> Using Default Prompt:\n{prompt_text}")

    print(f">>> Generating {args.num_samples} samples...")

    for i in range(args.num_samples):
        start = time.time()

        input_patches_list = patchilizer.encode_generate(prompt_text)
        input_patches = torch.tensor([input_patches_list], dtype=torch.long).to(device)

        try:
            generated_patches_list = model.generate(
                input_patches,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            raw_text = patchilizer.decode(generated_patches_list)

            # 你原版只做了这一步:contentReference[oaicite:2]{index=2}
            # 现在升级成结构化重写：metadata 锁死、正文强清洗
            final_abc = compose_final_abc(prompt_text, raw_text)

        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        abc_filename = f"{model_filename}_s{i}_{timestamp}.abc"
        save_path = os.path.join(output_dir, abc_filename)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_abc)

        print(f"[{i + 1}/{args.num_samples}] Saved ABC: {save_path}")

        run_abc2xml(save_path)

        print(f"Time: {time.time() - start:.2f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="llama")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)

    # 默认收敛一点，减少乱符号
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)

    args = parser.parse_args()
    generate_music(args)

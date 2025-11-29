import os
import time
import torch
import sys

# 路径 Hack，确保能导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Patchilizer
from config import *
from models.configuration import PiCoGenConfig
from models.student import PiCoGen

# 假设这里是 Baseline 训练出的纯 Student 权重
STUDENT_WEIGHTS_PATH = "weights/student_baseline.pth"  # 请修改为你实际的路径

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs(ORIGINAL_OUTPUT_FOLDER, exist_ok=True)
patchilizer = Patchilizer()


def load_student_model():
    print("[Inference] Loading PiCoGen Student Backbone...")
    config = PiCoGenConfig(
        backbone_type="llama",  # 需与训练时一致
        vocab_size=128,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=12,  # 需与训练时一致
        num_attention_heads=16,
        max_position_embeddings=PATCH_LENGTH
    )
    model = PiCoGen(config)

    if os.path.exists(STUDENT_WEIGHTS_PATH):
        ckpt = torch.load(STUDENT_WEIGHTS_PATH, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("Student weights loaded successfully.")
    else:
        print(f"Warning: Student weights not found at {STUDENT_WEIGHTS_PATH}!")

    return model.to(device)


model = load_student_model()
model.eval()


def inference_backbone(prompt_lines=[], pieces=NUM_SAMPLES):
    """
    Backbone 独立推理：Patch-Level Autoregressive
    """
    print("Starting Inference (Backbone Only)...")
    file_no = 1
    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

    while file_no <= pieces:
        start_time = time.time()

        # 1. 处理 Prompt
        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        # 记录已生成的字符以便保存
        byte_list = list(''.join(prompt_lines))

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        # Input: [1, Seq_Len, Patch_Size]
        input_patches = torch.tensor(prompt_patches, device=device).unsqueeze(0)

        # 2. 生成 (调用 student.py 的 generate)
        # 注意: 这会返回完整的 patches 列表 List[List[int]]
        try:
            generated_patches = model.generate(
                input_patches,
                max_length=PATCH_LENGTH // PATCH_SIZE,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P
            )
        except Exception as e:
            print(f"Generation Error: {e}")
            break

        # 3. 解码
        # generated_patches 包含了 Prompt，我们需要解码全部内容
        decoded_text = patchilizer.decode(generated_patches)

        # 4. 保存
        filename = time.strftime("%Y%m%d-%H%M%S") + f"_backbone_{file_no}.abc"
        save_path = os.path.join(ORIGINAL_OUTPUT_FOLDER, filename)

        with open(save_path, 'w') as w:
            w.write(decoded_text)

        print(f"Generated {filename}")
        file_no += 1


if __name__ == '__main__':
    inference_backbone()
import os
import time
import torch
import sys
from transformers import GPT2Config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Patchilizer
from config import *
from models.configuration import PiCoGenConfig
from models.student import PiCoGen
from models.teacher import NotaGenLMHeadModel
from models.wrapper import NotaGenStudentWrapper

# 配置权重路径
DISTILLED_STUDENT_PATH = "weights/student_distilled.pth"
TEACHER_CKPT_PATH = "weights/notagen_teacher.pth"  # 或者是合并了 LoRA 的 Teacher

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs(ORIGINAL_OUTPUT_FOLDER, exist_ok=True)
patchilizer = Patchilizer()


def load_distilled_system():
    print("[Inference] Loading Distilled System (Student + TeacherDecoder)...")

    # 1. 加载 Teacher (为了获取 CharDecoder)
    t_patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, max_length=PATCH_LENGTH, n_embd=HIDDEN_SIZE,
                                vocab_size=1)
    t_byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, max_length=PATCH_SIZE + 1, hidden_size=HIDDEN_SIZE,
                               vocab_size=128)
    teacher = NotaGenLMHeadModel(encoder_config=t_patch_config, decoder_config=t_byte_config)

    # 加载 Teacher 权重 (确保路径正确)
    if os.path.exists(TEACHER_CKPT_PATH):
        ckpt = torch.load(TEACHER_CKPT_PATH, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        teacher.load_state_dict(state_dict, strict=False)
    else:
        print("Error: Teacher weights not found! Decoder will be random.")

    # 2. 加载 Student (经过蒸馏的)
    s_config = PiCoGenConfig(
        backbone_type="llama",
        vocab_size=128,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=12,
        num_attention_heads=16,
        max_position_embeddings=PATCH_LENGTH
    )
    student = PiCoGen(s_config)

    if os.path.exists(DISTILLED_STUDENT_PATH):
        ckpt = torch.load(DISTILLED_STUDENT_PATH, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        student.load_state_dict(state_dict, strict=False)
        print("Distilled Student loaded.")
    else:
        print(f"Error: Distilled Student weights not found at {DISTILLED_STUDENT_PATH}")

    # 3. 组合 (Wrapper)
    # 这将使 generate 调用变成: Student(Encoder) -> Teacher(Decoder)
    model = NotaGenStudentWrapper(student, teacher.char_level_decoder, device)
    return model.to(device)


model = load_distilled_system()
model.eval()


def inference_distilled(prompt_lines=[], pieces=NUM_SAMPLES):
    """
    Distilled 推理：使用 NotaGen 的 Hierarchical 生成逻辑
    """
    print("Starting Inference (Distilled / NotaGen Style)...")
    file_no = 1
    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

    while file_no <= pieces:
        start_time = time.time()

        # Prompt 处理
        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        print(''.join(byte_list), end='')

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

        end_flag = False

        # 循环生成 Patch
        while True:
            # === 这里调用的是 Wrapper 的 generate，即 Student -> TeacherDec ===
            predicted_patch = model.generate(input_patches.unsqueeze(0),
                                             top_k=TOP_K,
                                             top_p=TOP_P,
                                             temperature=TEMPERATURE)

            # 停止条件判断
            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                end_flag = True
                break

            next_patch = patchilizer.decode([predicted_patch])
            for char in next_patch:
                byte_list.append(char)
                print(char, end='')

            # 拼接
            predicted_patch = torch.tensor([predicted_patch], device=device)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)

            # 长度保护
            if len(byte_list) > 102400: break

        # 保存
        filename = time.strftime("%Y%m%d-%H%M%S") + f"_distilled_{file_no}.abc"
        save_path = os.path.join(ORIGINAL_OUTPUT_FOLDER, filename)
        with open(save_path, 'w') as w:
            w.write(''.join(byte_list))

        print(f"\nSaved {filename}")
        file_no += 1


if __name__ == '__main__':
    inference_distilled()
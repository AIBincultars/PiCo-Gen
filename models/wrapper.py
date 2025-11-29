# models/wrapper.py
import torch
import torch.nn as nn
import numpy as np
from config import PATCH_SIZE
from samplings import top_k_sampling, top_p_sampling, temperature_sampling


def safe_normalize_probs(probs):
    epsilon = 1e-12
    probs = np.array(probs, dtype=np.float64)
    probs = np.where(np.isnan(probs) | (probs < 0), 0, probs)
    probs = probs + epsilon
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs[0] = 1.0
    return probs


class NotaGenStudentWrapper(nn.Module):
    """
    将 Student (Encoder) 和 Teacher (Char Decoder) 组合，
    模拟 NotaGenLMHeadModel 的行为。
    """

    def __init__(self, student_model, char_decoder, device):
        super().__init__()
        self.student = student_model
        self.char_level_decoder = char_decoder
        self.device = device
        self.bos_token_id = 1

    def generate(self, patches, top_k=0, top_p=1, temperature=1.0):
        # 1. 准备 Char 生成的上下文 tokens
        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:, :, -(patches.shape[-1] % PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:, :, :-(patches.shape[-1] % PATCH_SIZE)]
        else:
            tokens = torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, PATCH_SIZE)

        # 2. [关键] 使用 Student 生成 Hidden States (Latents)
        # Student 必须经过蒸馏，输出的 projected feature 才能被 Char Decoder 理解
        masks = torch.ones(patches.shape[0], patches.shape[1], device=self.device)
        student_out = self.student(patches, masks)
        encoded_patches = student_out['projected']  # [B, S, H]

        # 3. 使用 Teacher 的 Char Decoder 自回归生成 Patch 内的字符
        generated_patch = []
        while True:
            # 取最后一个 Patch 的 Embedding
            current_patch_embed = encoded_patches[0][-1]

            prob = self.char_level_decoder.generate(current_patch_embed, tokens).cpu().detach().numpy()
            prob = safe_normalize_probs(prob)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = safe_normalize_probs(prob)
            token = temperature_sampling(prob, temperature=temperature)

            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)

        return generated_patch
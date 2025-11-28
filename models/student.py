import torch
import torch.nn as nn
from .base import GenericStudent
from .configuration import PiCoGenConfig
from samplings import top_k_sampling, top_p_sampling, temperature_sampling


class PiCoGen(GenericStudent):
    def __init__(self, config: PiCoGenConfig):
        super().__init__(config, backbone_type=config.backbone_type)

    def forward(self, patches, masks):
        return super().forward(patches, masks)

    @torch.no_grad()
    def generate(self, input_patches, max_length=1024, temperature=1.0, top_k=0, top_p=0.9):
        """
        Patch-level Autoregressive Generation
        input_patches: [1, Seq_Len, Patch_Size]
        """
        self.eval()
        current_patches = input_patches.clone()

        # 假设 0 是 pad, 2 是 eos (根据你的 Tokenizer 设定)
        eos_token_id = 2

        for _ in range(max_length - current_patches.size(1)):
            # 1. 构造 Mask (全 1)
            masks = torch.ones(current_patches.size(0), current_patches.size(1), device=current_patches.device)

            # 2. Forward
            outputs = self.forward(current_patches, masks)
            logits = outputs['logits']  # [B, S, P, V]

            # 3. 取最后一个 Step 的预测
            next_patch_logits = logits[:, -1, :, :]  # [B, P, V] (这里假设 Batch=1)

            # 4. 采样 (对 Patch 内的每个 Token 独立/并行采样)
            # 这里的形状是 [Patch_Size, Vocab_Size]
            next_patch_tokens = []

            # 遍历 Patch 中的每一个位置 (0~15)
            for i in range(next_patch_logits.size(1)):
                token_logits = next_patch_logits[:, i, :]  # [B, V]
                probs = torch.softmax(token_logits, dim=-1).cpu().numpy()

                # 使用你提供的采样工具函数
                # 注意：sample 函数通常处理一维 array
                probs = top_k_sampling(probs[0], top_k=top_k, return_probs=True)
                probs = top_p_sampling(probs, top_p=top_p, return_probs=True)
                token = temperature_sampling(probs, temperature=temperature)

                next_patch_tokens.append(token)

            # 5. 拼接新生成的 Patch
            next_patch_tensor = torch.tensor([next_patch_tokens], device=current_patches.device).unsqueeze(
                0)  # [1, 1, 16]
            current_patches = torch.cat([current_patches, next_patch_tensor], dim=1)

            # 6. 简单的停止条件 (如果 Patch 中包含 EOS)
            if eos_token_id in next_patch_tokens:
                break

        return current_patches.squeeze(0).cpu().tolist()  # 返回 List[List[int]]
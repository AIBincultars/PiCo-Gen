# models/base.py
import torch
import torch.nn as nn
from .backbones.llama import LlamaBackbone
from .backbones.qwen import QwenBackbone
from .backbones.mamba import MambaBackbone
from .backbones.rwkv import RWKVBackbone
from .configuration import PiCoGenConfig  # 假设你的配置类在这里


class GenericStudent(nn.Module):
    def __init__(self, config: PiCoGenConfig, backbone_type="llama"):
        super().__init__()
        self.config = config
        self.backbone_type = backbone_type

        # 1. Patch Embedding (共享)
        # 输入: [Batch, Seq_Len, Patch_Size] -> [Batch, Seq_Len, Hidden_Size]
        self.patch_embedding = nn.Linear(config.patch_size * config.patch_vocab_size, config.hidden_size)

        # 2. Backbone Factory
        if backbone_type == "llama":
            self.backbone = LlamaBackbone(config)
        elif backbone_type == "qwen":
            self.backbone = QwenBackbone(config)
        elif backbone_type == "mamba":
            self.backbone = MambaBackbone(config)
        elif backbone_type == "rwkv":
            self.backbone = RWKVBackbone(config)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # 3. Char Head (用于 Baseline 训练和推理)
        self.char_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 4. Distill Projector (用于蒸馏)
        self.distill_proj = nn.Linear(config.hidden_size, config.teacher_hidden_size)

    def forward(self, patches, masks):
        # patches: [Batch, Seq_Len, Patch_Size]
        # masks: [Batch, Seq_Len]

        # One-hot encoding & Linear Embedding
        # 假设 patches 里的值已经是 0-127 的 token id
        patches_oh = torch.nn.functional.one_hot(patches, num_classes=self.config.patch_vocab_size).float()
        b, s, p, v = patches_oh.shape
        patches_flat = patches_oh.view(b, s, p * v)
        inputs_embeds = self.patch_embedding(patches_flat)  # [B, S, H]

        # Backbone Forward
        # 所有 backbone 都封装好了，只需要传 embeds 和 masks
        hidden_states = self.backbone(inputs_embeds, masks)

        # Heads
        logits = self.char_head(hidden_states)
        projected_latents = self.distill_proj(hidden_states)

        return {
            "logits": logits,
            "latents": hidden_states,
            "projected": projected_latents
        }
import torch
import torch.nn as nn
from .backbones.llama import LlamaBackbone
from .backbones.qwen import QwenBackbone
from .backbones.mamba import MambaBackbone
from .backbones.rwkv import RWKVBackbone
from .configuration import PiCoGenConfig


class GenericStudent(nn.Module):
    def __init__(self, config: PiCoGenConfig, backbone_type="llama"):
        super().__init__()
        self.config = config
        self.backbone_type = backbone_type

        # 1. Patch Embedding
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

        # 3. Char Head [CRITICAL FIX]
        # 输出维度改为 patch_size * vocab_size，以便并行预测 Patch 内的所有 Token
        self.char_head = nn.Linear(config.hidden_size, config.patch_size * config.vocab_size, bias=False)

        # 4. Distill Projector
        self.distill_proj = nn.Linear(config.hidden_size, config.teacher_hidden_size)

    def forward(self, patches, masks):
        # patches: [Batch, Seq_Len, Patch_Size] -> [B, S, 16]
        # masks: [Batch, Seq_Len]

        patches_oh = torch.nn.functional.one_hot(patches, num_classes=self.config.patch_vocab_size).float()
        b, s, p, v = patches_oh.shape
        patches_flat = patches_oh.view(b, s, p * v)  # [B, S, 16*128]

        inputs_embeds = self.patch_embedding(patches_flat)  # [B, S, Hidden]

        hidden_states = self.backbone(inputs_embeds, masks)

        logits = self.char_head(hidden_states)  # [B, S, 16*128]

        # Reshape logits to [B, S, Patch_Size, Vocab_Size] for easier loss calculation
        logits = logits.view(b, s, self.config.patch_size, self.config.patch_vocab_size)

        projected_latents = self.distill_proj(hidden_states)

        return {
            "logits": logits,
            "latents": hidden_states,
            "projected": projected_latents
        }
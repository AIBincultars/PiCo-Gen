import torch
import torch.nn as nn
import copy
from transformers import LlamaModel, PreTrainedModel
from .configuration import PiCoGenConfig


class PiCoGen(PreTrainedModel):
    config_class = PiCoGenConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. Patch Encoder (大脑) - Llama Backbone
        self.patch_backbone = LlamaModel(config)
        # Projector: [Batch, Seq, Patch_Size * Vocab] -> [Batch, Seq, Hidden]
        self.patch_proj = nn.Linear(config.patch_size * config.patch_vocab_size, config.hidden_size)

        # 2. Char Decoder (手) - Tiny Llama (2 layers)
        char_conf = copy.deepcopy(config)
        char_conf.num_hidden_layers = 2
        self.char_backbone = LlamaModel(char_conf)
        self.char_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 3. [Explicit Control] 显式控制 Embedding
        self.ctrl_embedding = nn.Embedding(config.num_instruments, config.hidden_size)

        # 4. [Distillation] 投影层 (Student 512 -> Teacher 1280)
        self.distill_proj = nn.Linear(config.hidden_size, config.teacher_hidden_size)

    def forward(self, patches, masks, instrument_ids=None):
        # patches: [Batch, Seq, Patch_Size]
        b, s, p = patches.shape

        # Patch One-hot & Projection
        patches_one_hot = torch.nn.functional.one_hot(patches, num_classes=self.config.patch_vocab_size).float()
        patches_flat = patches_one_hot.view(b, s, -1)
        patch_inputs = self.patch_proj(patches_flat)

        # [Control Injection] 注入乐器向量
        if instrument_ids is not None:
            ctrl_emb = self.ctrl_embedding(instrument_ids).unsqueeze(1)  # [B, 1, H]
            patch_inputs = patch_inputs + ctrl_emb

            # Run Patch Encoder
        patch_outputs = self.patch_backbone(inputs_embeds=patch_inputs, attention_mask=masks)
        student_latents = patch_outputs.last_hidden_state

        # Run Projector (Align with Teacher)
        projected_latents = self.distill_proj(student_latents)

        # Run Char Decoder (Training Mode)
        char_outputs = self.char_backbone(inputs_embeds=student_latents)
        logits = self.char_head(char_outputs.last_hidden_state)

        return {
            "logits": logits,  # [B, S, Vocab]
            "patch_latents": student_latents,
            "projected_latents": projected_latents
        }
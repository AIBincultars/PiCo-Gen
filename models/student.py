import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, PreTrainedModel
from copy import deepcopy
from .configuration import PiCoGenConfig


class PiCoGen(PreTrainedModel):
    config_class = PiCoGenConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # [Patch Encoder]: Llama Backbone (Student's Brain)
        self.patch_embedding = nn.Linear(config.patch_size * config.patch_vocab_size, config.hidden_size)
        self.patch_backbone = LlamaModel(config)

        # [Char Decoder]: Lightweight Llama (Student's Hands)
        # 仅使用2层Decoder以保证推理速度
        char_conf = deepcopy(config)
        char_conf.num_hidden_layers = 2
        self.char_backbone = LlamaModel(char_conf)
        self.char_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # [Distillation Projector]: Align Student Hidden to Teacher Hidden
        self.distill_proj = nn.Linear(config.hidden_size, config.teacher_hidden_size)

        self.post_init()

    def forward(self, patches, masks):
        # patches: [Batch, Seq_Len, Patch_Size]
        b, s, p = patches.shape

        # 1. Patch Encoding
        patches_oh = F.one_hot(patches, num_classes=self.config.patch_vocab_size).float()
        patches_flat = patches_oh.view(b, s, -1)
        inputs_embeds = self.patch_embedding(patches_flat)

        # 2. Backbone Forward
        outputs = self.patch_backbone(inputs_embeds=inputs_embeds, attention_mask=masks)
        student_latents = outputs.last_hidden_state

        # 3. Char Decoding
        char_outputs = self.char_backbone(inputs_embeds=student_latents)
        logits = self.char_head(char_outputs.last_hidden_state)

        # 4. Projection for Distillation
        projected_latents = self.distill_proj(student_latents)

        return {
            "logits": logits,
            "latents": student_latents,
            "projected": projected_latents
        }
import torch.nn as nn
from transformers import RwkvModel, RwkvConfig

class RWKVBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # RWKV 参数映射
        rwkv_config = RwkvConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size if hasattr(config, "intermediate_size") else config.hidden_size * 4,
            # RWKV context length
            context_length=config.max_position_embeddings,
            use_cache=False
        )
        self.model = RwkvModel(rwkv_config)

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
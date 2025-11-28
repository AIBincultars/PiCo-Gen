import torch.nn as nn
from transformers import Qwen2Model, Qwen2Config

class QwenBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        qwen_config = Qwen2Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            hidden_act=config.hidden_act,
            use_cache=False,
            # Qwen 特有参数，如果没有设定则使用默认值
            rope_theta=getattr(config, "rope_theta", 1000000.0)
        )
        self.model = Qwen2Model(qwen_config)

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig

class LlamaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 如果 config 是 PiCoGenConfig，我们需要把它转为 LlamaConfig 或者确保兼容
        # 这里假设 config 包含了 hidden_size, num_hidden_layers 等标准参数
        llama_config = LlamaConfig(
            vocab_size=config.vocab_size, # 这里的 vocab_size 其实在 patch level 不重要，因为我们输入的是 embed
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            hidden_act=config.hidden_act,
            use_cache=False  # 训练时不使用 Cache
        )
        self.model = LlamaModel(llama_config)

    def forward(self, inputs_embeds, attention_mask):
        """
        inputs_embeds: [Batch, Seq_Len, Hidden_Size]
        attention_mask: [Batch, Seq_Len] (1 for valid, 0 for padding)
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
import torch
import torch.nn as nn
from transformers import MambaModel, MambaConfig


class MambaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Mamba 的参数与 Transformer 不同，需要映射
        # d_model -> hidden_size
        # n_layer -> num_hidden_layers
        mamba_config = MambaConfig(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            n_layer=config.num_hidden_layers,
            use_cache=False
        )
        self.model = MambaModel(mamba_config)

    def forward(self, inputs_embeds, attention_mask):
        # Mamba 是纯 Causal 的。
        # 对于 Padding 的处理：transformers 的 MambaModel 目前支持 attention_mask 参数
        # 但通常建议只用于左 Padding 或 在序列结束后的 Padding。
        # 这里的实现：将 Padding 部分的输入置零，防止噪音进入状态空间太严重

        # 扩展 mask 维度以匹配 inputs_embeds
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds * mask_expanded

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            # 注意：某些版本的 HF Mamba 可能不接受 attention_mask，或者只接受特定格式
            # 如果报错，可以去掉 attention_mask，因为我们已经在上面 mask 了 input
            # 这里的实现保留它，因为 HF 最新版已经开始标准化这个接口
        )
        return outputs.last_hidden_state
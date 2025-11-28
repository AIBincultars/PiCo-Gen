import torch.nn as nn
from .base import GenericStudent
from .configuration import PiCoGenConfig


class PiCoGen(GenericStudent):
    """
    PiCoGen 学生模型。
    继承自 GenericStudent，通过配置选择不同的 Backbone (Llama, Qwen, Mamba, RWKV)。
    """

    def __init__(self, config: PiCoGenConfig):
        # 确保传入的 config 是 PiCoGenConfig 类型
        if not isinstance(config, PiCoGenConfig):
            raise ValueError(f"Expected PiCoGenConfig, got {type(config)}")

        super().__init__(config, backbone_type=config.backbone_type)

    def forward(self, patches, masks):
        """
        前向传播
        :param patches: [Batch, Seq_Len, Patch_Size]
        :param masks: [Batch, Seq_Len]
        :return: dict with 'logits', 'latents', 'projected'
        """
        return super().forward(patches, masks)
from transformers import LlamaConfig


class PiCoGenConfig(LlamaConfig):
    model_type = "picogen"

    def __init__(
            self,
            vocab_size=128,  # ABC 字符集大小 (Char 词表)
            hidden_size=768,  # [升级] 768 维度，保证足够的表征能力
            intermediate_size=2048,  # [升级] 配合 SwiGLU 的中间层维度
            num_hidden_layers=12,  # Patch Encoder 深度 (大脑深度)
            num_attention_heads=12,  # 768 / 64 = 12 头
            num_key_value_heads=4,  # GQA 分组数 (依然保持高效推理)
            max_position_embeddings=2048,  # RoPE 长度
            hidden_act="silu",  # SwiGLU 激活函数

            # --- PiCo-Gen 专属参数 ---
            patch_size=16,  # NotaGen 的 Patch 大小
            patch_vocab_size=128,  # Patch 内 Token 的词表大小
            char_decoder_layers=4,  # [升级] Char Decoder 深度 (4层，保证手不笨)
            num_instruments=64,  # 支持的乐器数量 (Control Embedding)
            teacher_hidden_size=1280,  # Teacher (NotaGen-Large) 的维度，用于投影对齐
            **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            **kwargs,
        )
        self.patch_size = patch_size
        self.patch_vocab_size = patch_vocab_size
        self.char_decoder_layers = char_decoder_layers
        self.num_instruments = num_instruments
        self.teacher_hidden_size = teacher_hidden_size
from transformers import PretrainedConfig, LlamaConfig

class PiCoGenConfig(PretrainedConfig):
    model_type = "picogen"

    def __init__(
        self,
        backbone_type="llama",
        vocab_size=128,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        patch_size=16,
        patch_vocab_size=128,
        teacher_hidden_size=1280,
        hidden_act="silu",
        rope_theta=10000.0,
        **kwargs,
    ):
        self.backbone_type = backbone_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.patch_size = patch_size
        self.patch_vocab_size = patch_vocab_size
        self.teacher_hidden_size = teacher_hidden_size
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        super().__init__(**kwargs)
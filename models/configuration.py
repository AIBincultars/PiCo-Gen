from transformers import LlamaConfig
import config


class PiCoGenConfig(LlamaConfig):
    model_type = "picogen"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 从 config.py 读取配置
        self.vocab_size = config.VOCAB_SIZE
        self.hidden_size = config.STUDENT_HIDDEN_SIZE
        self.intermediate_size = config.STUDENT_INTERMEDIATE_SIZE
        self.num_hidden_layers = config.STUDENT_NUM_LAYERS
        self.num_attention_heads = config.STUDENT_NUM_HEADS
        self.num_key_value_heads = config.STUDENT_NUM_KV_HEADS
        self.max_position_embeddings = config.STUDENT_MAX_POS

        # 自定义参数
        self.patch_size = config.PATCH_SIZE
        self.patch_vocab_size = config.VOCAB_SIZE
        self.num_instruments = config.NUM_INSTRUMENTS
        self.teacher_hidden_size = config.TEACHER_HIDDEN_SIZE
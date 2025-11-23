import os

# ================= 路径配置 =================
DATA_RAW_PATH = "data/raw"
DATA_INDEX_PATH = "data/processed/train.jsonl"
# 这里填你下载好的 NotaGen-X 权重路径
TEACHER_WEIGHTS_PATH = "weights/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
STUDENT_SAVE_PATH = "weights/picogen_stage1.pth"

# ================= 模型参数 =================
# [Student: Tiny-Llama]
STUDENT_HIDDEN_SIZE = 512
STUDENT_LAYERS = 12
STUDENT_HEADS = 16
STUDENT_KV_HEADS = 4  # GQA
INTERMEDIATE_SIZE = 1376

# [Teacher: NotaGen-Large]
TEACHER_HIDDEN_SIZE = 1280
TEACHER_LAYERS = 20
TEACHER_HEADS = 20

# [Common]
PATCH_SIZE = 16
VOCAB_SIZE = 128
NUM_INSTRUMENTS = 64

# ================= 训练参数 =================
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
TEMPERATURE = 2.0

# ================= Loss 权重 =================
ALPHA_KD = 2.0        # 蒸馏权重 (学习老师)
BETA_STRUCT = 0.5     # 结构权重 (学习大脑)
GAMMA_PHYSICS = 0.2   # 物理权重 (自我修养)
LAMBDA_ENTROPY = 0.05 # 熵权重 (防止重复)

# ================= 环境 =================
DEVICE = "cuda"
WANDB_LOGGING = False
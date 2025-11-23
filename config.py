import os

# ==========================================
# 1. 路径配置 (Paths)
# ==========================================
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed/train.jsonl"
TEACHER_WEIGHTS_PATH = "weights/notagen_large.pth"  # 你的 Teacher 权重路径
STUDENT_SAVE_PATH = "weights/picogen_student.pth"
LOGS_PATH = "logs/training.log"

# ==========================================
# 2. 数据与架构参数 (Model Architecture)
# ==========================================
# 基础配置
PATCH_SIZE = 16
PATCH_LENGTH = 1024
VOCAB_SIZE = 128           # ABC 字符集大小
NUM_INSTRUMENTS = 64       # 支持的乐器数量 (控制显式 Embedding)

# Teacher 配置 (NotaGen-Large)
TEACHER_HIDDEN_SIZE = 1280
TEACHER_PATCH_LAYERS = 20
TEACHER_CHAR_LAYERS = 6

# Student 配置 (Tiny-Llama-Music)
STUDENT_HIDDEN_SIZE = 512
STUDENT_INTERMEDIATE_SIZE = 1376  # SwiGLU 维度
STUDENT_NUM_LAYERS = 12           # 深度
STUDENT_NUM_HEADS = 16
STUDENT_NUM_KV_HEADS = 4          # GQA 分组数 (4组)
STUDENT_MAX_POS = 2048            # RoPE 长度

# ==========================================
# 3. 训练超参数 (Training Hyperparams)
# ==========================================
BATCH_SIZE = 8            # 5070Ti 建议 8-16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
ACCUMULATION_STEPS = 1
TEMPERATURE = 2.0         # 蒸馏温度

# ==========================================
# 4. Loss 权重配置 (Loss Weights)
# ==========================================
ALPHA_KD = 2.0            # Logits 蒸馏权重
BETA_STRUCT = 0.5         # 结构(Hidden State)蒸馏权重
GAMMA_PHYSICS = 0.2       # 物理(Power Law) Loss 权重
LAMBDA_ENTROPY = 0.05     # 熵正则化权重 (防止重复)
LAMBDA_SEMANTIC = 0.1     # 语义(CLaMP) Loss 权重 (可选)

# ==========================================
# 5. 硬件与环境 (Environment)
# ==========================================
DEVICE = "cuda"
WANDB_LOGGING = False
WANDB_KEY = ""
EXP_TAG = "PiCoGen_v1"
import os

# ===================== 数据 =====================
# 直接指向你合并后的索引；先用已转好的集(如 POP909)，后续把 Lakh/Groove/E-GMD 追加到 joint 即可
# 多个 jsonl 或 目录都行；用分号 ; 分隔
DATA_TRAIN_INDEX_PATH = "/mnt/d/project/NotaGen-main/data/POP909/abc_aug_train.jsonl"
DATA_EVAL_INDEX_PATH  = "/mnt/d/project/NotaGen-main/data/POP909/abc_aug_eval.jsonl"

# 可选：与上面路径一一对应的采样权重（不写则均匀）
#DATA_TRAIN_WEIGHTS = "2,1,1,1"   # 例：POP909:Lakh:Groove:E-GMD = 2:1:1:1


# ===================== 预训练起点 =====================
# 官方/他人提供的 .pth 作为 Q-LoRA 微调的起点（FP32 加载 → 4bit 量化 → 注入 LoRA）
PRETRAINED_PATH = "weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
LOAD_PRETRAINED = False               # True=热启动（仅加载模型权重，不加载优化器）
# 若要从你自己上次训练的 ckpt 续训，则把 LOAD_FROM_CHECKPOINT=True，且确保 WEIGHTS_PATH 指向你的 ckpt

# ===================== 模型规模 =====================
PATCH_STREAM     = True              # 流式编码（保持不变）
PATCH_SIZE       = 16
PATCH_LENGTH     = 1024              # 16GB 显存下 1024 比较稳；想更长先测再加
CHAR_NUM_LAYERS  = 6
PATCH_NUM_LAYERS = 20
HIDDEN_SIZE      = 1280

# ===================== Q-LoRA 相关（训练脚本会读取这些） =====================
USE_BNB_4BIT        = True           # 使用 bitsandbytes 4bit 量化（nf4）
BNB_4BIT_QUANT_TYPE = "nf4"          # nf4 更稳
BNB_COMPUTE_DTYPE   = "bfloat16"     # 计算用 bf16
BNB_DOUBLE_QUANT    = True           # 双重量化（更省显存）

# LoRA 超参（建议值；显存紧张时 r=16）
QLORA_LORA_R        = 32
QLORA_LORA_ALPHA    = 32
QLORA_LORA_DROPOUT  = 0.05

# ===================== 训练超参 =====================
# —— 16GB 稳妥档：1×16 的有效 batch（能跑通更重要）
BATCH_SIZE            = 1
ACCUMULATION_STEPS    = 16           # 有效 batch = 1*16
LEARNING_RATE         = 3e-4         # Q-LoRA 可取稍大的 LR；并入新库时降到 1e-4 ~ 6e-5
NUM_EPOCHS            = 5           # 先 20~30 epoch 观察；满意再延长
PATCH_SAMPLING_BATCH_SIZE = 0

# ===================== 断点续训 / 日志 =====================
LOAD_FROM_CHECKPOINT = False         # 只有在续训你自己的 ckpt 时才设 True
WANDB_LOGGING        = False
WANDB_KEY            = "<your_wandb_key>"

# ===================== 命名与输出 =====================
EXP_TAG = "qlora_sft"
NAME =  EXP_TAG + \
        "_p_size_"   + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_c_layers_" + str(CHAR_NUM_LAYERS) + \
        "_h_size_"   + str(HIDDEN_SIZE) + \
        "_lr_"       + str(LEARNING_RATE) + \
        "_batch_"    + str(BATCH_SIZE)

WEIGHTS_PATH = "checkpoints/weights_notagen_" + NAME + ".pth"   # 你自己的 ckpt 会保存到这里
LOGS_PATH    = "logs/logs_notagen_"          + NAME + ".txt"
WANDB_NAME   = NAME

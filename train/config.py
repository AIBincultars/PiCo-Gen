import os

# Configuration for the data
DATA_TRAIN_INDEX_PATH = "data/preprocessed/SymphonyNet_dataset/symphonynet_augmented_train.jsonl"
DATA_EVAL_INDEX_PATH  = "data/preprocessed/SymphonyNet_dataset/symphonynet_augmented_eval.jsonl"

# Configuration for the model
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
CHAR_NUM_LAYERS = 6                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                           # Number of layers in the encoder
HIDDEN_SIZE = 1280                                              # Hidden Size

# Configuration for the training
BATCH_SIZE = 1         
LEARNING_RATE = 1e-5   
NUM_EPOCHS = 64                                                 # Number of epochs to train for
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full context
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
WANDB_LOGGING = False                                           # Whether to log to wandb
WANDB_KEY = '<your_wandb_key>'

# --- [新增] 为了防止 main.py 报错必须定义的变量 ---
# 即使是 Baseline 模式，这些变量也必须存在，但可以是空值
TEACHER_CKPT = None                                             # Baseline 模式下不使用，设为 None 即可
ALPHA_KD = 0.0                                                  # 蒸馏权重
BETA_PHY = 0.0                                                  # 物理约束权重

PRETRAINED_PATH = ""                 # Path of pretrained weights
EXP_TAG = ''                                                    # Experiment tag for name differentiation
NAME =  EXP_TAG + \
        "_p_size_" + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_c_layers_" + str(CHAR_NUM_LAYERS) + \
        "_h_size_" + str(HIDDEN_SIZE) + \
        "_lr_" + str(LEARNING_RATE) + \
        "_batch_" + str(BATCH_SIZE)

WEIGHTS_PATH = "weights_notagen_" + NAME + ".pth"                   # Path to save weights
LOGS_PATH    = "logs_notagen_"    + NAME + ".txt"                      # Path to save logs
WANDB_NAME = NAME
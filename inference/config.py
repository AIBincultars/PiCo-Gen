import os

# Configuration for the model (必须与 train/config.py 保持一致)
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
CHAR_NUM_LAYERS = 6                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                           # Number of layers in the encoder
HIDDEN_SIZE = 1280                                              # Hidden Size

# Configuration for inference (推理专用参数)
INFERENCE_WEIGHTS_PATH = ''                                     # Path to weights for inference
NUM_SAMPLES = 100                                               # Number of samples to generate
TOP_K = 9                                                       # Top k for sampling
TOP_P = 0.9                                                     # Top p for sampling
TEMPERATURE = 1.2                                               # Temperature for sampling

# Output Paths (根据参数自动生成路径)
base_name = os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0]
folder_suffix = f'_k_{TOP_K}_p_{TOP_P}_temp_{TEMPERATURE}'

ORIGINAL_OUTPUT_FOLDER = os.path.join('../output/original', base_name + folder_suffix)
INTERLEAVED_OUTPUT_FOLDER = os.path.join('../output/interleaved', base_name + folder_suffix)
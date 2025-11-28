# 1. 确保在主目录下
# 2. 启动训练 (Baseline 模式, 使用 Llama 骨干)
python main.py \
  --mode baseline \
  --backbone llama \
  --exp_name llama_v1 \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 20
python generate.py \
  --weights_path ../checkpoints/llama_v1/weights_picogen_llama_v1_llama_lr0.0001_bz4_best.pth \
  --num_samples 1 \
  --max_length 1024 \
  --temperature 1.0 \
  --top_k 40 \
  --top_p 0.95 \
  --prompt "generate a song"
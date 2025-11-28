import argparse
import os
import torch
import train.config as train_cfg
import inference.config as inf_cfg
from train.engine import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="PiCo-Gen Experiment Platform")

    # --- 实验控制 ---
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "distill"], help="实验模式")
    parser.add_argument("--backbone", type=str, required=True, choices=["llama", "qwen", "mamba", "rwkv"],
                        help="学生模型骨干")
    parser.add_argument("--exp_name", type=str, default="debug_run", help="实验标识名")

    # --- 路径配置 ---
    parser.add_argument("--save_dir", type=str, default="weights/", help="权重保存路径")
    parser.add_argument("--train_data", type=str, default=train_cfg.DATA_INDEX_PATH, help="训练数据路径")
    parser.add_argument("--teacher_ckpt", type=str, default=train_cfg.TEACHER_CKPT,
                        help="Teacher权重路径(仅distill模式需要)")

    # --- 训练超参 (默认值读取自 train/config.py) ---
    parser.add_argument("--batch_size", type=int, default=train_cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=train_cfg.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=train_cfg.NUM_EPOCHS)

    # --- Loss 权重 ---
    parser.add_argument("--alpha_kd", type=float, default=train_cfg.ALPHA_KD)
    parser.add_argument("--beta_phy", type=float, default=train_cfg.BETA_PHY)

    args = parser.parse_args()

    # 简单检查
    if args.mode == "distill" and (not args.teacher_ckpt or not os.path.exists(args.teacher_ckpt)):
        print(f"Warning: Teacher checkpoint '{args.teacher_ckpt}' not found or not provided for distillation.")

    print(f"=== Starting Experiment: {args.exp_name} ===")
    print(f"Mode: {args.mode} | Backbone: {args.backbone}")

    runner = ExperimentRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
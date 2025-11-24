# -*- coding: utf-8 -*-
import argparse
import os
from training.engine import run_training_engine


def main():
    parser = argparse.ArgumentParser(description="PiCo-Gen Training Launcher")

    # --- 路径配置 ---
    parser.add_argument("--exp_name", type=str, default="picogen_v1", help="实验名称，用于保存文件命名")
    parser.add_argument("--save_dir", type=str, default="weights/picogen", help="权重保存目录")
    parser.add_argument("--train_data", type=str, default="data/symphonynet_processed_train.jsonl",
                        help="训练数据JSONL路径")
    parser.add_argument("--teacher_ckpt", type=str,
                        default="weights/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth",
                        help="Teacher模型权重路径")

    # --- 模型超参 ---
    parser.add_argument("--t_hidden", type=int, default=1280, help="Teacher hidden size (NotaGen-X default: 1280)")
    parser.add_argument("--s_hidden", type=int, default=512, help="Student hidden size (TinyLlama)")
    parser.add_argument("--s_layers", type=int, default=12, help="Student layer count")

    # --- 训练超参 ---
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)

    # --- 损失权重 (消融实验关键参数) ---
    parser.add_argument("--alpha_hid", type=float, default=2.0, help="Hidden State蒸馏权重")
    parser.add_argument("--beta_phy", type=float, default=0.5, help="物理感知损失权重 (消融实验设为0.0)")
    parser.add_argument("--phy_smooth", type=float, default=0.5, help="物理损失内部：平滑度权重")
    parser.add_argument("--phy_ent", type=float, default=0.1, help="物理损失内部：熵权重")

    args = parser.parse_args()

    # 确保目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 启动训练引擎
    run_training_engine(args)


if __name__ == "__main__":
    main()
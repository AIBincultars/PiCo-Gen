import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, get_cosine_schedule_with_warmup

# Import from our modules
from models.teacher import NotaGenLMHeadModel
from models.student import PiCoGen, PiCoGenConfig
from training.dataset import SymphonyDataset, collate_fn
from training.losses import PhysicsAwareLoss


def run_training_engine(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting PiCo-Gen Experiment: {args.exp_name} ===")
    print(f"Device: {device} | Phy Loss: {args.beta_phy}")

    # -------------------------
    # 1. Prepare Models
    # -------------------------
    # Teacher (NotaGen-X)
    print(">>> Loading Teacher...")
    t_patch_cfg = GPT2Config(n_embd=args.t_hidden, n_layer=20, n_head=20, n_positions=1024, vocab_size=1)
    t_char_cfg = GPT2Config(n_embd=args.t_hidden, n_layer=6, n_head=20, vocab_size=128)
    teacher = NotaGenLMHeadModel(t_patch_cfg, t_char_cfg)

    if os.path.exists(args.teacher_ckpt):
        ckpt = torch.load(args.teacher_ckpt, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        teacher.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Teacher ckpt {args.teacher_ckpt} not found! Using random weights.")

    teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False

    # Student (PiCoGen)
    print(">>> Initializing Student...")
    s_config = PiCoGenConfig(
        vocab_size=128, hidden_size=args.s_hidden, num_hidden_layers=args.s_layers,
        teacher_hidden_size=args.t_hidden, patch_size=16
    )
    student = PiCoGen(s_config).to(device)

    # -------------------------
    # 2. Data & Optimizer
    # -------------------------
    train_ds = SymphonyDataset(args.train_data, patch_len=1024)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    optimizer = AdamW(student.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, args.epochs * len(train_loader))
    phy_loss_fn = PhysicsAwareLoss(lambda_smooth=args.phy_smooth, lambda_entropy=args.phy_ent)

    # -------------------------
    # 3. Training Loop
    # -------------------------
    student.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0

        for step, (patches, masks) in enumerate(pbar):
            patches, masks = patches.to(device), masks.to(device)

            # Teacher Forward
            with torch.no_grad():
                t_patches = patches.reshape(len(patches), -1, 16)
                # 注意: 这里需确保 teacher.patch_level_decoder 返回正确
                t_enc_out = teacher.patch_level_decoder(t_patches, masks)
                t_latents = t_enc_out["last_hidden_state"]
                t_logits = teacher(patches, masks)  # Logits for Physics Loss

            # Student Forward
            s_out = student(patches, masks)
            s_logits = s_out['logits']
            s_proj = s_out['projected']

            # Losses
            loss_hidden = F.mse_loss(s_proj, t_latents)
            loss_phy, _ = phy_loss_fn(s_logits, t_logits)

            # KD Loss (Logits)
            temp = 2.0
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / temp, dim=-1),
                F.softmax(t_logits / temp, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)

            total_loss = loss_kd + args.alpha_hid * loss_hidden + args.beta_phy * loss_phy

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({"L_Total": f"{total_loss.item():.3f}", "L_Phy": f"{loss_phy.item():.3f}"})

        # Checkpoint
        save_path = os.path.join(args.save_dir, f"{args.exp_name}_ep{epoch + 1}.pth")
        torch.save(student.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")
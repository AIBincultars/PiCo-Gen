# -*- coding: utf-8 -*-
import os
import gc
import time
import math
import json
import wandb
import weakref
import torch
import random
import unicodedata
import re
import numpy as np
from abctoolkit.transpose import Key2index, Key2Mode
from utils import *
from config import *
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, LlamaConfig, get_scheduler, get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# === Q-LoRA ===
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------- ASCII 清洗，避免奇怪字符触发 CUDA/Embedding 错 ----------
try:
    from unidecode import unidecode
except Exception:
    def unidecode(x): return x

ASCII_ALLOWED = set("\t\n\r" + "".join(chr(i) for i in range(32, 127)))
TRANS_MAP = {
    0xFEFF:"", 0x00A0:" ", 0x2018:"'", 0x2019:"'",
    0x201C:'"', 0x201D:'"', 0x2013:"-", 0x2014:"-",
    0x2212:"-", 0x2026:"...", 0x00B7:"*",
}
def sanitize_abc(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).translate(str.maketrans(TRANS_MAP))
    s = re.sub(r"[\u200B-\u200D\u2060\uFEFF]", "", s)
    s = unidecode(s)
    s = "".join(ch if ch in ASCII_ALLOWED else "?" for ch in s)
    return s.replace("\r\n","\n").replace("\r","\n")

# ---------- QLoRA 可从 config.py 覆盖的超参（无则用默认） ----------
try: USE_BNB_4BIT
except NameError: USE_BNB_4BIT = True
try: BNB_4BIT_QUANT_TYPE
except NameError: BNB_4BIT_QUANT_TYPE = "nf4"
try: BNB_COMPUTE_DTYPE
except NameError: BNB_COMPUTE_DTYPE = "bfloat16"
try: BNB_DOUBLE_QUANT
except NameError: BNB_DOUBLE_QUANT = True

try: QLORA_LORA_R
except NameError: QLORA_LORA_R = 32
try: QLORA_LORA_ALPHA
except NameError: QLORA_LORA_ALPHA = 32
try: QLORA_LORA_DROPOUT
except NameError: QLORA_LORA_DROPOUT = 0.05

# ---------- dtype 解析 ----------
_compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(BNB_COMPUTE_DTYPE, torch.bfloat16)

Index2Key = {index: key for key, index in Key2index.items() if index not in [1, 11]}
Mode2Key = {mode: key for key, mode_list in Key2Mode.items() for mode in mode_list }

# ========== DDP ==========
world_size = int(os.environ.get('WORLD_SIZE', 1))
global_rank = int(os.environ.get('RANK', 0))
local_rank  = int(os.environ.get('LOCAL_RANK', 0))

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl')
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ========== Seeds ==========
seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = BATCH_SIZE
patchilizer = Patchilizer()

# ========== Model Config ==========
patch_config = GPT2Config(
    num_hidden_layers=PATCH_NUM_LAYERS,
    max_length=PATCH_LENGTH,
    max_position_embeddings=PATCH_LENGTH,
    n_embd=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE//64,
    vocab_size=1
)
char_config = GPT2Config(
    num_hidden_layers=CHAR_NUM_LAYERS,
    max_length=PATCH_SIZE+1,
    max_position_embeddings=PATCH_SIZE+1,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE//64,
    vocab_size=128
)

# ========== Build Base Model (FP32) ==========
model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config).to(device)
# --- HF/PEFT forward 兼容层：允许 input_ids / attention_mask 关键字 ---
if not hasattr(NotaGenLMHeadModel, "_notagen_orig_forward"):
    # 保存原始 forward
    model._notagen_orig_forward = model.forward

    def _compat_forward(self, *args, **kwargs):
        """
        兼容两种调用：
        1) self(original): forward(input_patches, input_masks)
        2) HF/PEFT: forward(input_ids=..., attention_mask=..., **kwargs)
        """
        if ("input_ids" in kwargs) or ("attention_mask" in kwargs):
            input_patches = kwargs.get("input_ids",  args[0] if len(args) > 0 else None)
            input_masks   = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            return self._notagen_orig_forward(input_patches, input_masks)
        # 保持原有位置参数调用
        return self._notagen_orig_forward(*args, **kwargs)

    # 将兼容 forward 绑定到类（所有实例生效）
    setattr(NotaGenLMHeadModel, "forward", _compat_forward)
# --- /兼容层 ---

# ---- 先加载 PRETRAINED_PATH（FP32），再做 4bit & LoRA ----
def load_pretrained_fp32(model, path):
    if not path or not os.path.exists(path):
        print(f"[PRETRAINED] skip: {path}")
        return model
    sd = torch.load(path, map_location="cpu")
    state = sd["model"] if (isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict)) else sd
    # 清理 module. 前缀
    cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[PRETRAINED] loaded with missing={len(missing)} unexpected={len(unexpected)} from {path}")
    return model

if not LOAD_FROM_CHECKPOINT:
    model = load_pretrained_fp32(model, PRETRAINED_PATH)

# ---- 兼容不同 bitsandbytes 版本的 Linear4bit 构造 ----
def _make_linear4bit(in_features, out_features, bias, compute_dtype, quant_type):
    # bnb 0.48.x 使用 input_features/output_features；更高版本也接受这个名字
    try:
        return bnb.nn.Linear4bit(
            input_features=in_features,
            output_features=out_features,
            bias=bias,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            compress_statistics=True,
            quant_storage=torch.uint8,
        )
    except TypeError:
        # 若你的 bnb 版本使用 in_features/out_features（极少见），退回旧参数名
        return bnb.nn.Linear4bit(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            compress_statistics=True,
            quant_storage=torch.uint8,
        )

def linear_to_4bit_with_weight(linear: torch.nn.Linear) -> bnb.nn.Linear4bit:
    target_device = linear.weight.device

    # 先创建，再立刻迁移到目标设备（此时还没写入 uint8 权重）
    new_lin = _make_linear4bit(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=(linear.bias is not None),
        compute_dtype=_compute_dtype,
        quant_type=BNB_4BIT_QUANT_TYPE,
    ).to(target_device)

    # 量化原 FP32 权重（返回 uint8 + quant_state）
    qweight, quant_state = bnb.functional.quantize_4bit(
        linear.weight.data.to(torch.float32).contiguous(),
        quant_type=BNB_4BIT_QUANT_TYPE,
        compress_statistics=True,
    )

    # 搬到同一设备（有些版本 quant_state 不支持 .to()，做容错）
    qweight = qweight.to(target_device)
    try:
        quant_state = quant_state.to(target_device)
    except Exception:
        pass

    # 写入量化后的权重（此后不要再调用 new_lin.to(...)）
    new_lin.weight = bnb.nn.Params4bit(
        qweight, requires_grad=False, quant_state=quant_state
    )

    # 拷贝 bias 到同一设备（不替换 Parameter 对象）
    if linear.bias is not None and new_lin.bias is not None:
        with torch.no_grad():
            new_lin.bias.data.copy_(linear.bias.data.to(target_device))

    return new_lin





def replace_linear_with_4bit_inplace(module: torch.nn.Module):
    for name, child in list(module.named_children()):
        replace_linear_with_4bit_inplace(child)
        if isinstance(child, torch.nn.Linear) and USE_BNB_4BIT:
            setattr(module, name, linear_to_4bit_with_weight(child))


replace_linear_with_4bit_inplace(model)

# ---- 准备 k-bit 训练 & 注入 LoRA（只训练 LoRA 参数） ----
model = prepare_model_for_kbit_training(model)

# ===================== 变动①：补一个 no-op 的 prepare_inputs_for_generation =====================
if not hasattr(NotaGenLMHeadModel, "prepare_inputs_for_generation"):
    def _pifg(self, *args, **kwargs):
        # 训练阶段不会用到；仅为满足 PEFT 在包装时的属性访问
        return kwargs
    setattr(NotaGenLMHeadModel, "prepare_inputs_for_generation", _pifg)
# ===================== 变动① 结束 =====================

def collect_lora_targets(m: torch.nn.Module):
    preferred = ["q_proj","k_proj","v_proj","o_proj","c_attn","c_proj","c_fc","fc_in","fc_out","out_proj","in_proj","wq","wk","wv","wo"]
    linear_types = (torch.nn.Linear, bnb.nn.Linear4bit)
    suffixes = []
    for full, mod in m.named_modules():
        if isinstance(mod, linear_types):
            # ===================== 变动②：过滤 lm_head / embedding 层 =====================
            low = full.lower()
            if ("lm_head" in low) or ("embed" in low) or ("embedding" in low):
                continue
            # ===================== 变动② 结束 =====================
            suffixes.append(full.split(".")[-1])
    suffixes = list(dict.fromkeys(suffixes))
    targets = [s for s in suffixes if s in preferred] or (suffixes or ["c_attn","c_proj","c_fc","fc_in","fc_out"])
    return targets

target_modules = collect_lora_targets(model)
lora_cfg = LoraConfig(
    r=QLORA_LORA_R,
    lora_alpha=QLORA_LORA_ALPHA,
    lora_dropout=QLORA_LORA_DROPOUT,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# DDP
if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# AMP & Optim
scaler = GradScaler(enabled=True)
is_autocast = True

optimizer = bnb.optim.PagedAdamW8bit(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

def clear_unused_tensors():
    gc.disable()
    try:
        model_tensors = {id(p) for p in (model.module.parameters() if hasattr(model, "module") else model.parameters())}
        optimizer_tensors = {
            id(state)
            for state_dict in optimizer.state.values()
            for state in state_dict.values()
            if isinstance(state, torch.Tensor)
        }
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
        tensor_refs = [weakref.ref(tensor) for tensor in tensors]
        for tensor_ref in tensor_refs:
            tensor = tensor_ref()
            if tensor is not None and id(tensor) not in model_tensors and id(tensor) not in optimizer_tensors:
                tensor.detach_()
                del tensor
    except:
        pass
    finally:
        gc.enable()
        gc.collect()
        torch.cuda.empty_cache()

def collate_batch(input_batches):
    input_patches, input_masks = zip(*input_batches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks  = torch.nn.utils.rnn.pad_sequence(input_masks,  batch_first=True, padding_value=0)
    return input_patches.to(device), input_masks.to(device)

def split_into_minibatches(input_patches, input_masks, minibatch_size):
    mb = int(minibatch_size) if minibatch_size is not None else 1
    if mb <= 0:
        mb = 1
    n = len(input_patches)
    if mb >= n:
        # 不需要再切，直接返回一个小批
        return [(input_patches, input_masks)]
    minibatches = []
    for start_idx in range(0, n, mb):
        end_idx = start_idx + mb
        minibatch_patches = input_patches[start_idx:end_idx]
        minibatch_masks  = input_masks[start_idx:end_idx]
        minibatches.append((minibatch_patches, minibatch_masks))
    return minibatches


class NotaGenDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        filepath = self.filenames[idx]['path']
        ori_key = Mode2Key[self.filenames[idx]['key']]

        # 依据原 key 周围 ±3 半音的分布随机转调
        ori_key_index  = Key2index[ori_key]
        available_index = [(ori_key_index + offset) % 12 for offset in range(-3, 4)]
        index_prob      = [1/16, 2/16, 3/16, 4/16, 3/16, 2/16, 1/16]
        index_prob_rng  = [0] + [sum(index_prob[0:i+1]) for i in range(len(index_prob))]
        rnd = random.random()
        for i in range(len(index_prob_rng)-1):
            if index_prob_rng[i] <= rnd < index_prob_rng[i+1]:
                des_key_index = available_index[i]
                break
        if des_key_index == 1:
            des_key = 'Db' if random.random() < 0.8 else 'C#'
        elif des_key_index == 11:
            des_key = 'B' if random.random() < 0.8 else 'Cb'
        elif des_key_index == 6:
            des_key = 'F#' if random.random() < 0.5 else 'Gb'
        else:
            des_key = Index2Key[des_key_index]

        # 识别相对路径的根目录（来自 jsonl 所在目录）
        item_root = self.filenames[idx].get("__root__", "")

        # 拆分原始路径与文件名
        folder = os.path.dirname(filepath)
        base   = os.path.splitext(os.path.basename(filepath))[0]
        ext    = os.path.splitext(os.path.basename(filepath))[1] or ".abc"

        def _abs(p):
            # 若 p 是绝对路径，直接返回
            if os.path.isabs(p):
                return os.path.normpath(p)
            # 有 root 的情况下，避免重复拼出 ".../POP909/POP909/..."
            if item_root:
                bn = os.path.basename(os.path.normpath(item_root))  # 例如 "POP909"
                p_norm = p.replace("\\", "/")
                if p_norm.startswith(bn + "/") or p_norm.startswith(bn + os.sep):
                    p = p[len(bn)+1:]  # 去掉重复的前缀 "POP909/"
                return os.path.normpath(os.path.join(item_root, p))
            return os.path.normpath(p)


        # 优先：离线转调 <key>/<name>_<key>.abc
        # 回退：原始 <name>.abc
        # 最后：索引中的 path（自动补后缀）
        candidates = [
            _abs(os.path.join(folder, des_key, f"{base}_{des_key}{ext}")),
            _abs(os.path.join(folder, f"{base}{ext}")),
            _abs(filepath if os.path.splitext(filepath)[1] else f"{filepath}{ext}"),
        ]

        des_filepath = None
        for p in candidates:
            if os.path.exists(p):
                des_filepath = p
                break
        if des_filepath is None:
            # 兜底：在 root 下对 base*.abc 做一次模糊匹配
            import glob
            g = glob.glob(_abs(os.path.join(folder, f"{base}*.abc")))
            if g:
                des_filepath = g[0]

        if des_filepath is None:
            raise FileNotFoundError(f"ABC file not found. Tried: {candidates}")

        with open(des_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            abc_text = f.read()

        des_filepath = None
        for p in candidates:
            if os.path.exists(p):
                des_filepath = p
                break
        if des_filepath is None:
            # 作为最后一次兜底，尝试模糊匹配 base*.abc
            import glob
            g = glob.glob(os.path.join(folder, f"{base}*.abc"))
            if g:
                des_filepath = g[0]

        if des_filepath is None:
            raise FileNotFoundError(
                f"ABC file not found. Tried: {candidates}"
            )

        with open(des_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            abc_text = f.read()

        abc_text = sanitize_abc(abc_text)  # 不注入 prompt，只做 ASCII 清洗

        file_bytes = patchilizer.encode_train(abc_text)
        file_masks = [1] * len(file_bytes)
        return torch.tensor(file_bytes, dtype=torch.long), torch.tensor(file_masks, dtype=torch.long)

def process_one_batch(batch):
    input_patches, input_masks = batch
    with autocast(dtype=torch.bfloat16):
        loss = model(input_patches, input_masks).loss
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)
    return loss

def train_epoch(epoch):
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = (epoch-1)*len(train_set)

    for batch in tqdm_train_set:
        micro = max(1, BATCH_SIZE // max(1, ACCUMULATION_STEPS))
        minibatches = split_into_minibatches(batch[0], batch[1], micro)
        for minibatch in minibatches:
            loss = process_one_batch(minibatch) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            total_train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        train_steps += 1
        if global_rank==0 and WANDB_LOGGING:
            wandb.log({"train_loss": total_train_loss / iter_idx}, step=train_steps)
        iter_idx += 1
        if iter_idx % 1000 == 0:
            clear_unused_tensors()
    return total_train_loss / (iter_idx-1)

def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
    for batch in tqdm_eval_set:
        micro = max(1, BATCH_SIZE // max(1, ACCUMULATION_STEPS))
        minibatches = split_into_minibatches(batch[0], batch[1], micro)
        for minibatch in minibatches:
            with torch.no_grad(), autocast(dtype=torch.bfloat16):
                loss = process_one_batch(minibatch) / ACCUMULATION_STEPS
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    return total_eval_loss / (iter_idx-1)

# ========== Train & Eval ==========
if __name__ == "__main__":
    if WANDB_LOGGING and global_rank==0:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="notagen", name=WANDB_NAME)

    # --- ensure log/ckpt parent dirs exist ---
    def _ensure_parent(p):
        d = os.path.dirname(p)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    _ensure_parent(LOGS_PATH)
    _ensure_parent(WEIGHTS_PATH)
    # --- /ensure ---

    # load data
    # --------- NEW: 多索引混采支持（分号分隔，带可选权重） ---------
    def _split_paths(s):
        # 路径可为 jsonl 文件或目录；分号 ; 分隔多个
        paths = []
        for p in [x.strip() for x in s.split(";") if x.strip()]:
            if os.path.isdir(p):
                for fn in os.listdir(p):
                    if fn.endswith(".jsonl"):
                        paths.append(os.path.join(p, fn))
            else:
                paths.append(p)
        return paths


    def _read_jsonl(fp):
        items = []
        root = os.path.dirname(fp)  # 记录当前索引文件所在目录
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    # 给每条样本打上来源根目录，供 Dataset 解析相对路径
                    if isinstance(obj, dict):
                        obj.setdefault("__root__", root)
                    items.append(obj)
                except:
                    pass
        return items



    def _load_multi_jsonl(path_field: str, weight_field: str = None, is_eval=False):
        path_list = _split_paths(path_field)
        if not path_list:
            raise ValueError("No jsonl paths found. Check DATA_*_INDEX_PATH in config.py")

        # 解析权重（与路径一一对应；不填则均匀）
        if weight_field and isinstance(weight_field, str) and weight_field.strip():
            ws = [w.strip() for w in weight_field.split(",") if w.strip()]
            if len(ws) == len(path_list):
                weights = [max(1, int(float(w))) for w in ws]
            else:
                print("[WARN] DATA_TRAIN_WEIGHTS length != number of paths. Using uniform weights.")
                weights = [1] * len(path_list)
        else:
            weights = [1] * len(path_list)

        pools = []
        for p in path_list:
            if not os.path.exists(p):
                print(f"[WARN] Missing index path: {p}")
                pools.append([])
                continue
            items = _read_jsonl(p)
            random.shuffle(items)
            pools.append(items)
            print(f"[INDEX] loaded {len(items)} from {p}")

        if is_eval:
            merged = []
            for items in pools:
                merged.extend(items)
            return merged

        total_len = sum(len(x) for x in pools)
        if total_len == 0:
            return []

        # 简单“加权轮询”混采
        def cyc(items):
            while True:
                yield random.choice(items)

        cyc_iters = [cyc(items) if len(items) > 0 else None for items in pools]
        mixed, target_len = [], total_len
        while len(mixed) < target_len:
            for k, w in enumerate(weights):
                if len(mixed) >= target_len:
                    break
                if cyc_iters[k] is None:
                    continue
                for _ in range(w):
                    mixed.append(next(cyc_iters[k]))
                    if len(mixed) >= target_len:
                        break
        return mixed
    # --------- NEW: 本地划分函数（eval 为空时使用） ---------
    def _auto_split(items, eval_ratio=0.05):
        items = list(items)
        random.shuffle(items)
        n = len(items)
        k = max(1, int(n * eval_ratio)) if n > 0 else 0
        eval_part  = items[:k]
        train_part = items[k:]
        return train_part, eval_part
    # --------- /NEW ---------


    # --------- /NEW ---------

    print("Loading Data...")
    train_files = _load_multi_jsonl(
        DATA_TRAIN_INDEX_PATH,
        weight_field=globals().get("DATA_TRAIN_WEIGHTS", None),
        is_eval=False
    )
    eval_files = _load_multi_jsonl(
        DATA_EVAL_INDEX_PATH,
        weight_field=None,
        is_eval=True
    )

    if len(eval_files) == 0:
        train_files, eval_files = _auto_split(train_files, eval_ratio=0.05)

    train_batch_nums = int(len(train_files) / batch_size)
    eval_batch_nums  = int(len(eval_files) / batch_size)

    random.shuffle(train_files); random.shuffle(eval_files)
    train_files = train_files[:train_batch_nums*batch_size]
    eval_files  = eval_files[:eval_batch_nums*batch_size]

    train_set = NotaGenDataset(train_files)
    eval_set  = NotaGenDataset(eval_files)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
    eval_sampler  = DistributedSampler(eval_set,  num_replicas=world_size, rank=local_rank)

    train_set = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler, shuffle=(train_sampler is None))
    eval_set  = DataLoader(eval_set,  batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler,  shuffle=(eval_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000)

    # ====== Checkpoint / Pretrained 逻辑 ======
    if LOAD_FROM_CHECKPOINT:
        if os.path.exists(WEIGHTS_PATH):
            checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
            if torch.cuda.device_count() > 1 and hasattr(model, "module"):
                cpu_model = deepcopy(model.module)
                cpu_model.load_state_dict(checkpoint['model'], strict=False)
                model.module.load_state_dict(cpu_model.state_dict(), strict=False)
            else:
                cpu_model = deepcopy(model)
                cpu_model.load_state_dict(checkpoint['model'], strict=False)
                model.load_state_dict(cpu_model.state_dict(), strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print("Optimizer state load failed. Re-init. Err:", e)
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_sched'])
            except Exception as e:
                print("LR scheduler state load failed. Re-init. Err:", e)
            pre_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            min_eval_loss = checkpoint['min_eval_loss']
            print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
            checkpoint = None
        else:
            raise Exception('Checkpoint not found to continue training. Please check your parameter settings.')
    else:
        # 新开微调（已在上面加载了 PRETRAINED_PATH 并完成 QLoRA 注入）
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = 100.0

    for epoch in range(1+pre_epoch, NUM_EPOCHS+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch(epoch)
        eval_loss  = eval_epoch()
        if global_rank==0:
            # ——双保险：确保父目录存在
            _ensure_parent(WEIGHTS_PATH)
            _ensure_parent(LOGS_PATH)

            # 1) 先保存更优的 checkpoint
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = {
                    'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'min_eval_loss': min_eval_loss
                }
                # 原子化保存：先写临时文件，再替换
                tmp_path = WEIGHTS_PATH + ".tmp"
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, WEIGHTS_PATH)
                checkpoint = None  # 释放内存

            # 2) 再写日志；即使失败，也不影响已保存的 ckpt
            try:
                with open(LOGS_PATH,'a', encoding='utf-8') as f:
                    f.write(
                        "Epoch " + str(epoch) +
                        "\ntrain_loss: " + str(train_loss) +
                        "\neval_loss: " + str(eval_loss) +
                        "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n"
                    )
            except Exception as e:
                # 兜底：尝试创建父目录后重试一次
                try:
                    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
                    with open(LOGS_PATH,'a', encoding='utf-8') as f:
                        f.write(
                            "Epoch " + str(epoch) +
                            "\ntrain_loss: " + str(train_loss) +
                            "\neval_loss: " + str(eval_loss) +
                            "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n"
                        )
                except Exception as e2:
                    print(f"[WARN] write log failed: {e2}")

        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Min Eval Loss : "+str(min_eval_loss))

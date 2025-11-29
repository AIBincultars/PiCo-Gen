import os
import json
import torch
from torch.utils.data import Dataset
from data.tokenizer import Patchilizer


class SymphonyDataset(Dataset):
    """
    用于 Backbone 预训练或蒸馏的数据集（通常只包含 ABC 内容）。
    """

    def __init__(self, jsonl_path, patch_len=1024):
        self.data = []
        self.patchilizer = Patchilizer()
        self.patch_len = patch_len

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Index file not found: {jsonl_path}")

        print(f"[Dataset] Loading index from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    pass
        print(f"[Dataset] Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = item.get('path', '')

        # 智能路径修正
        if path and not os.path.exists(path):
            alt_paths = [os.path.join('data', path), path.replace('preprocessed/', 'data/preprocessed/')]
            for p in alt_paths:
                if os.path.exists(p):
                    path = p
                    break

        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.patch_len, dtype=torch.long), torch.zeros(self.patch_len, dtype=torch.long)

        patches = self.patchilizer.encode_train(text, patch_length=self.patch_len)
        patches = torch.tensor(patches, dtype=torch.long)
        masks = torch.ones(len(patches), dtype=torch.long)
        return patches, masks


class InstructionDataset(Dataset):
    """
    [新增] 专门用于 LoRA 指令微调的数据集。
    处理格式: {"path": "...", "instruction": "...", ...}
    """

    def __init__(self, jsonl_path, patch_len=1024):
        self.data = []
        self.patchilizer = Patchilizer()
        self.patch_len = patch_len

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Instruction file not found: {jsonl_path}")

        print(f"[InstructionDataset] Loading from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"[InstructionDataset] Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 获取指令
        instruction = item.get('instruction', '')

        # 2. 获取 ABC 内容 (通过 path 读取)
        path = item.get('path', '')
        abc_content = ""

        # 路径容错逻辑
        if path and not os.path.exists(path):
            # 尝试拼接常见前缀
            alt_paths = [os.path.join('data', path), os.path.join('../data', path)]
            for p in alt_paths:
                if os.path.exists(p):
                    path = p
                    break

        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    abc_content = f.read()
            except Exception as e:
                print(f"[Warning] Error reading file {path}: {e}")

        # 3. 构造模型输入
        # 格式示例: Instruction \n ABC_Content
        # NotaGen 通常将 Prompt 作为 Metadata 处理，或者直接接在前面
        full_text = f"{instruction}\n{abc_content}"

        # 4. 编码
        patches = self.patchilizer.encode_train(full_text, patch_length=self.patch_len)
        patches = torch.tensor(patches, dtype=torch.long)
        masks = torch.ones(len(patches), dtype=torch.long)

        return patches, masks


def collate_fn(batch):
    patches, masks = zip(*batch)
    patches = torch.nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return patches, masks
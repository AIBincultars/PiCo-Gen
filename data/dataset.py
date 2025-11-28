import os
import json
import torch
from torch.utils.data import Dataset
from data.tokenizer import Patchilizer


class SymphonyDataset(Dataset):
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

        # --- [关键修改] 智能路径修正逻辑 ---
        path = item['path']

        # 1. 如果路径不存在，尝试拼接 'data/' 前缀
        if not os.path.exists(path):
            alt_path = os.path.join('data', path)
            if os.path.exists(alt_path):
                path = alt_path
            else:
                # 还可以尝试相对于 jsonl 文件所在的目录 (如果 jsonl 也在 data/ 下)
                # 这种容错机制能避免大多数 FileNotFoundError
                print(f"[Warning] File not found: {item['path']} (tried {alt_path})")
        # -------------------------------------

        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 使用 tokenizer 处理文本
            patches = self.patchilizer.encode_train(text, patch_length=self.patch_len)
            patches = torch.tensor(patches, dtype=torch.long)
            masks = torch.ones(len(patches), dtype=torch.long)
            return patches, masks

        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Fallback: 返回随机数据防止训练中断，但会打印错误
            return torch.randint(0, 128, (self.patch_len,)), torch.ones(self.patch_len)


def collate_fn(batch):
    patches, masks = zip(*batch)
    patches = torch.nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return patches, masks
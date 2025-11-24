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

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except:
                    pass
        print(f"[Dataset] Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # 这里的 path 是绝对路径，或者相对于项目根目录的路径
            with open(item['path'], 'r', encoding='utf-8') as f:
                text = f.read()
            patches = self.patchilizer.encode_train(text, patch_length=self.patch_len)
            patches = torch.tensor(patches, dtype=torch.long)
            masks = torch.ones(len(patches), dtype=torch.long)
            return patches, masks
        except Exception as e:
            print(f"Error loading {item.get('path', 'unknown')}: {e}")
            # Fallback
            return torch.randint(0, 128, (self.patch_len,)), torch.ones(self.patch_len)


def collate_fn(batch):
    patches, masks = zip(*batch)
    patches = torch.nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return patches, masks
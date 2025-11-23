import json
import torch
from torch.utils.data import Dataset
import config
from data.tokenizer import Patchilizer  # 假设你已经搬运好了


class PiCoGenDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.patchilizer = Patchilizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            with open(item['path'], 'r', encoding='utf-8') as f:
                abc_text = f.read()

            # 使用 Patchilizer 编码 (调用 encode_train)
            # 注意：这里仅作为示例，具体调用参数需参考原 utils.py
            patches = self.patchilizer.encode_train(abc_text, patch_length=config.PATCH_LENGTH)

            # 转换为 Tensor
            patches_tensor = torch.tensor(patches, dtype=torch.long)
            masks_tensor = torch.ones(len(patches), dtype=torch.long)
            inst_id_tensor = torch.tensor(item['inst_id'], dtype=torch.long)

            return {
                "patches": patches_tensor,
                "masks": masks_tensor,
                "inst_ids": inst_id_tensor
            }
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            # 返回一个 dummy 数据防止报错
            return self.__getitem__((idx + 1) % len(self.data))


def collate_fn(batch):
    # 简单 Padding
    patches = [item['patches'] for item in batch]
    masks = [item['masks'] for item in batch]
    inst_ids = torch.stack([item['inst_ids'] for item in batch])

    patches_padded = torch.nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0)
    masks_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)

    return {
        "patches": patches_padded,
        "masks": masks_padded,
        "inst_ids": inst_ids
    }
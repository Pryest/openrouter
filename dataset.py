from torch.utils.data import Dataset
import torch
import os
from utils import stream_jsonl


class RouterCollectionDataset(Dataset):
    def __init__(self, data_path, use_emb=False):
        super().__init__()
        self.data = []
        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                if not files:
                    continue
                for file in files:
                    if file.endswith(".jsonl"):
                        if use_emb:
                            embs = torch.load(os.path.join(root, file.replace(".jsonl", ".pt")))
                        for i, item in enumerate(stream_jsonl(os.path.join(root, file))):
                            if use_emb:
                                item["logits"] = embs[i]["logits"]
                            self.data.append(item)
        else:
            assert data_path.endswith(".jsonl")
            if use_emb:
                embs = torch.load(data_path.replace(".jsonl", ".pt"))
            for i, item in enumerate(stream_jsonl(data_path)):
                if use_emb:
                    item["logits"] = embs[i]["logits"]
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


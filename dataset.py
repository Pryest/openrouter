from torch.utils.data import Dataset
import os
from utils import stream_jsonl


class RouterCollectionDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data = []
        for root, dirs, files in os.walk(data_dir):
            if not files:
                continue
            for file in files:
                if file.endswith(".jsonl"):
                    for item in stream_jsonl(os.path.join(root, file)):
                        self.data.append(item)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


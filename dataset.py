from torch.utils.data import Dataset
import os
from utils import stream_jsonl


class RouterCollectionDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []
        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                if not files:
                    continue
                for file in files:
                    if file.endswith(".jsonl"):
                        for item in stream_jsonl(os.path.join(root, file)):
                            self.data.append(item)
        else:
            assert data_path.endswith(".jsonl")
            for item in stream_jsonl(data_path):
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


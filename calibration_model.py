import torch
import os
from transformers import AutoModel


class CalibrationLinear(torch.nn.Module):
    def __init__(self, hidden_size):
        super(CalibrationLinear, self).__init__()
        self.wik = torch.nn.Linear(hidden_size, 2, dtype=torch.float32)
        self.wik.weight.data.fill_(0)
    
    def forward(self, hidden_states):
        return self.wik(hidden_states)
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.wik.state_dict(), os.path.join(path, "wik.pt"))
    
    def load(self, path):
        self.wik.load_state_dict(torch.load(os.path.join(path, "wik.pt")))


class CalibrationEncoder(torch.nn.Module):
    def __init__(self, load_from=None):
        super(CalibrationEncoder, self).__init__()
        if load_from is not None:
            self.model = AutoModel.from_pretrained(load_from)
            self.wik = torch.nn.Linear(self.model.config.hidden_size, 2, bias=False, dtype=self.model.dtype)
            self.wik.weight.data.fill_(0)
    
    def forward(self, *args, **kwargs):
        hidden_states = self.model(*args, **kwargs).last_hidden_state
        logits = self.wik(hidden_states)[:, 0]
        return logits
    
    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.wik.state_dict(), os.path.join(path, "wik.pt"))

    def load(self, path):
        self.model = AutoModel.from_pretrained(path)
        self.wik.load_state_dict(torch.load(os.path.join(path, "wik.pt")))
    

class CalibrationDecoder(torch.nn.Module):
    def __init__(self, load_from=None):
        super(CalibrationDecoder, self).__init__()
        if load_from is not None:
            self.model = AutoModel.from_pretrained(load_from, _attn_implementation="flash_attention_2", torch_dtype="bfloat16")
            self.wik = torch.nn.Linear(self.model.config.hidden_size, 2, bias=False, dtype=self.model.dtype)
            self.wik.weight.data.fill_(1)
    
    def forward(self, *args, **kwargs):
        hidden_states = self.model(*args, **kwargs).last_hidden_state
        logits = self.wik(hidden_states)[:, -1]
        return logits
    
    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.wik.state_dict(), os.path.join(path, "wik.pt"))
    
    def load(self, path):
        self.model = AutoModel.from_pretrained(path)
        self.wik.load_state_dict(torch.load(os.path.join(path, "wik.pt")))
    

import torch
import os
from transformers import AutoModel


class CalibrationLinear(torch.nn.Module):
    def __init__(self, hidden_size):
        super(CalibrationLinear, self).__init__()
        self.wik = torch.nn.Linear(hidden_size, 2, dtype=torch.float32, bias=False)
        torch.manual_seed(1001)
        torch.nn.init.normal_(self.wik.weight.data, mean=0.0, std=1)
        print(self.wik.weight.data, flush=True)
    
    def forward(self, hidden_states):
        return self.wik(hidden_states)
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.wik.state_dict(), os.path.join(path, "wik.pt"))
    
    def load(self, path):
        self.wik.load_state_dict(torch.load(os.path.join(path, "wik.pt")))


class CalibrationMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(CalibrationMLP, self).__init__()
        self.wik = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, intermediate_size, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(intermediate_size, 2, dtype=torch.float32)
        )
        torch.manual_seed(1001)
        torch.nn.init.normal_(self.wik[0].weight.data, mean=0.0, std=1)
        print(self.wik[0].weight.data, flush=True)
        torch.nn.init.normal_(self.wik[2].weight.data, mean=0.0, std=1)
        print(self.wik[2].weight.data, flush=True)

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
            self.model = AutoModel.from_pretrained(load_from).to(torch.float32)
            self.wik = torch.nn.Linear(self.model.config.hidden_size, 2, bias=False, dtype=torch.float32)
            torch.manual_seed(1001)
            torch.nn.init.normal_(self.wik.weight.data, mean=0.0, std=1)
            print(self.wik.weight.data, flush=True)
            
    
    def forward(self, *args, **kwargs):
        hidden_states = self.model(*args, **kwargs).last_hidden_state
        logits = self.wik(hidden_states[:, 0])
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
            if torch.cuda.is_available():
                self.model = AutoModel.from_pretrained(load_from, _attn_implementation="flash_attention_2", torch_dtype="bfloat16")
            else:
                self.model = AutoModel.from_pretrained(load_from, torch_dtype="bfloat16")
            self.wik = torch.nn.Linear(self.model.config.hidden_size, 2, bias=False, dtype=torch.float32)
            torch.manual_seed(1001)
            torch.nn.init.normal_(self.wik.weight.data, mean=0.0, std=1)
            print(self.wik.weight.data, flush=True)
    
    def forward(self, *args, **kwargs):
        hidden_states = self.model(*args, **kwargs).last_hidden_state
        logits = self.wik(hidden_states[:, -1])
        return logits
    
    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.wik.state_dict(), os.path.join(path, "wik.pt"))
    
    def load(self, path):
        self.model = AutoModel.from_pretrained(path)
        self.wik.load_state_dict(torch.load(os.path.join(path, "wik.pt")))
    
if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--ft_path", type=str, required=True)

    args = parser.parse_args()

    prefix = "/fs-computility/llmdelivery/$USER/ckpts/0108/"
    ft_path = args.ft_path
    print(ft_path)

    model_prefix = "/fs-computility/llm/shared/llmeval/models/opencompass_hf_hub/"

    local_models = {
        "llama-7b":f"{model_prefix}/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",

        "llama2-7b":f"{model_prefix}/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
        
        "llama3.2-1b":f"{model_prefix}/models--meta-llama--Llama-3.2-1B/snapshots/5d853ed7d16ac794afa8f5c9c7f59f4e9c950954",
        "llama3.2-3b":f"{model_prefix}/models--meta-llama--Llama-3.2-3B/snapshots/5cc0ffe09ee49f7be6ca7c794ee6bd7245e84e60",

        "mistral-v0.1-7b":f"{model_prefix}/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
        "mistral-v0.3-7b":f"{model_prefix}/models--mistralai--Mistral-7B-v0.3/snapshots/b67d6a03ca097c5122fa65904fce0413500bf8c8",
    }    

    ft_states = torch.load(ft_path, map_location="cpu")

    model = ft_path[len(prefix):-3] if ft_path.endswith(".pt") else ft_path[len(prefix):]
    pretrained_path = local_models["-".join((model.split("/")[0].split("-")[:-1]))]

    model = CalibrationDecoder(pretrained_path)
    
    infos = model.load_state_dict(ft_states, strict=False)

    print(infos)

    model.save(ft_path.replace(".pt", ""))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.save_pretrained(ft_path.replace(".pt", ""))
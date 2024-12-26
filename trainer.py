import torch

from dataset import RouterCollectionDataset

import os
from functools import partial

from accelerate import Accelerator

from torch.utils.data import DataLoader
from accelerate import PartialState
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# PartialState().is_main_process


def collate_fn(batch, tokenizer, is_hard):
    texts = []
    labels = []
    for item in batch:
        texts.append(item["prompt"])
        labels.append(item["passed"])
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128, pad_to_max_length=True)
    labels = torch.tensor(labels, dtype=torch.long if is_hard else torch.bfloat16)
    return inputs, labels


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_data_folder,
        save_path=None,
        train_batch_size=8,
        train_type="hard",
        eval_data_folder=None,
        eval_batch_size=8,
        eval_type="soft",
        micro_num=1,
        max_epochs=1,
        lr=1e-2,
        cosine_T_max=500,
        eval_every=50,
    ):
        self.accelerator = Accelerator(
            mixed_precision="bf16",
        )
    
        assert train_type in ["hard", "soft"]
        assert eval_type in ["hard", "soft"]

        self.train_type = train_type
        self.eval_type = eval_type
        self.micro_num = micro_num
        self.max_epochs = max_epochs

        datasets = {}
        dls = {}

        model = model.to(self.accelerator.device)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=cosine_T_max)
        
        self.model = self.accelerator.prepare(model)
        if optimizer is not None:
            self.optimizer = self.accelerator.prepare(optimizer)
        if scheduler is not None:
            self.scheduler = self.accelerator.prepare(scheduler)

        if train_data_folder:
            train_collate_fn = partial(collate_fn, tokenizer=tokenizer, is_hard=(train_type == "hard"))
            datasets["train"] = RouterCollectionDataset(train_data_folder)
            dls["train"] = DataLoader(datasets["train"], collate_fn=train_collate_fn, batch_size=train_batch_size, shuffle=True)
            self.train_dl = self.accelerator.prepare(dls["train"])
        else:
            self.train_dl = None

        if eval_data_folder:
            eval_collate_fn = partial(collate_fn, tokenizer=tokenizer, is_hard=(eval_type == "hard"))
            datasets["eval"] = RouterCollectionDataset(eval_data_folder)
            dls["eval"] = DataLoader(datasets["eval"], collate_fn=eval_collate_fn, batch_size=eval_batch_size, shuffle=False)
            self.eval_dl = self.accelerator.prepare(dls["eval"])
        else:
            self.eval_dl = None

        self.save_path = save_path
        self.eval_every = eval_every
        self.last_eval_losses = []


    def _step(self, batch, loss_type):
        inputs, labels = batch
        logits = self.model(**inputs)
        if loss_type == "hard":
            loss_list = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        else:
            loss_0_list = torch.nn.functional.cross_entropy(logits, torch.zeros(labels.size(), dtype=torch.long, device=logits.device), reduction="none")
            loss_1_list = torch.nn.functional.cross_entropy(logits, torch.ones(labels.size(), dtype=torch.long, device=logits.device), reduction="none")
            loss_list = labels * loss_1_list + (1 - labels) * loss_0_list
        
        return loss_list.mean()


    def save_model(self, special_name):

        model_state_dict = self.accelerator.get_state_dict(self.model)

        if PartialState().is_main_process:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(model_state_dict, os.path.join(self.save_path, special_name))
            print(f"Model successfully saved to {os.path.join(self.save_path, special_name)}")
    

    def eval(self, ckpt_name):
        assert self.eval_dl is not None

        if PartialState().is_main_process:
            print("Start evaluation.")
        
        with torch.no_grad():
            self.model.eval()

            loss_numerator = 0.
            loss_denominator = 0

            with tqdm(self.eval_dl, disable=True) as pbar:

                for batch in self.eval_dl:
                    loss = self._step(batch, self.eval_type)
                    loss_list = self.accelerator.gather_for_metrics(loss)

                    loss_numerator += loss_list.sum().item()
                    loss_denominator += len(loss_list)

                    # pbar.set_postfix_str(f"loss = {loss_numerator / loss_denominator}")
            
            if PartialState().is_main_process:
                print("Evaluation finished.")
                print(f"Final eval loss = {loss_numerator / loss_denominator} by {ckpt_name}", flush=True)
            
        return loss_numerator / loss_denominator


    def train_(self, max_epochs=1):
        
        step = 0
        loss_accumulate = 0

        for epoch in range(max_epochs):
            self.model.train()
            with tqdm(self.train_dl, disable=True) as pbar:
                with self.accelerator.accumulate(self.model):
                    for batch in pbar:
                        loss = self._step(batch, self.train_type)
                        loss_accumulate += loss.item()

                        self.accelerator.backward(loss)

                        step += 1
                        
                        if step % self.micro_num == 0:
                            if PartialState().is_main_process:
                                print(f"loss = {loss_accumulate / self.micro_num} at epoch {epoch} step {step}")
                            
                            loss_accumulate = 0
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()

                            if (step // self.micro_num) % self.eval_every == 0:
                                if self.eval_dl is not None:
                                    loss = self.eval(f"epoch_{epoch}_step_{step}.pt")
                                    self.last_eval_losses.append(loss)
                                    self.model.train()
                                self.save_model(f"epoch_{epoch}_step_{step}.pt")
                        
                    else:
                        if step % self.micro_num != 0:
                            pass
                            # drop last batch, elsewise:
                            # self.optimizer.step()
                            # self.scheduler.step()
                            # self.optimizer.zero_grad()

            self.accelerator.wait_for_everyone()

            if self.eval_dl is not None:
                loss = self.eval(f"epoch_{epoch}.pt")
                self.last_eval_losses.append(loss)
            
            self.save_model(f"epoch_{epoch}.pt")

        self.accelerator.end_training()


    def train(self):
        self.train_(self.max_epochs)

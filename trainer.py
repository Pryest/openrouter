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

from torch.utils.tensorboard import SummaryWriter
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
        tb_log_dir=None,
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
            datasets["eval"] = {}
            for root, dirs, files in os.walk(eval_data_folder):
                for file in files:
                    if file.endswith(".jsonl"):
                        datasets["eval"][file.replace(".jsonl", "")] = RouterCollectionDataset(os.path.join(root, file))
            
            dls["eval"] = {}
            self.eval_dl = {}
            for k, v in datasets["eval"].items():
                dls["eval"][k] = DataLoader(v, collate_fn=eval_collate_fn, batch_size=eval_batch_size, shuffle=False)
                self.eval_dl[k] = self.accelerator.prepare(dls["eval"][k])
        else:
            self.eval_dl = None

        self.save_path = save_path
        self.eval_every = eval_every
        self.last_eval_losses = []

        if tb_log_dir is not None and PartialState().is_main_process:
            os.makedirs(tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir, max_queue=5, purge_step=0, flush_secs=3)
        else:
            self.tb_writer = None


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
            eval_losses = {}

            for k, eval_dl in self.eval_dl.items():                
                loss_numerator = 0.
                loss_denominator = 0

                with tqdm(eval_dl, disable=True) as pbar:

                    for batch in eval_dl:
                        loss = self._step(batch, self.eval_type)
                        loss_list = self.accelerator.gather_for_metrics(loss)

                        loss_numerator += loss_list.sum().item()
                        loss_denominator += torch.ones_like(loss_list).sum().item()

                        # pbar.set_postfix_str(f"loss = {loss_numerator / loss_denominator}")

                eval_losses[k] = loss_numerator / loss_denominator

                self.accelerator.wait_for_everyone()

            if PartialState().is_main_process:
                print("Evaluation finished.")
                for k, v in eval_losses.items():
                    print(f"Eval loss/{k} = {v} by {ckpt_name}", flush=True)

        return eval_losses


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
                            
                            if self.tb_writer is not None:
                                self.tb_writer.add_scalar(tag="loss", scalar_value=loss_accumulate / self.micro_num, global_step=step)                                

                            loss_accumulate = 0
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()

                            if (step // self.micro_num) % self.eval_every == 0:
                                if self.eval_dl is not None:
                                    losses = self.eval(f"epoch_{epoch}_step_{step}.pt")
                                    if self.tb_writer is not None:
                                        for k, v in losses.items():
                                            self.tb_writer.add_scalar(tag=f"loss/{k}", scalar_value=v, global_step=step)
                                    self.last_eval_losses.append(losses)
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
                losses = self.eval(f"epoch_{epoch}.pt")
                if self.tb_writer is not None:
                    for k, v in losses.items():
                        self.tb_writer.add_scalar(tag=f"loss/{k}", scalar_value=v, global_step=step)
                self.last_eval_losses.append(losses)
            
            self.save_model(f"epoch_{epoch}.pt")

        self.accelerator.end_training()


    def train(self):
        self.train_(self.max_epochs)

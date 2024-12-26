from calibration_model import CalibrationDecoder
from transformers import AutoTokenizer
from trainer import Trainer
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--global_batch_size", type=int, default=7680)
    parser.add_argument("--micro_train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--t_max", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--tb_log_dir", type=str, default=None)

    args = parser.parse_args()

    model = CalibrationDecoder(args.model)
    model.model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, padding_side="left")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        save_path=args.save_folder,
        train_data_folder=args.train_data,
        eval_data_folder=args.eval_data,
        train_batch_size=args.micro_train_batch_size,
        eval_batch_size=args.eval_batch_size,
        micro_num=args.global_batch_size // args.micro_train_batch_size // 8,
        max_epochs=args.epochs,
        lr=args.learning_rate,
        cosine_T_max=args.t_max,
        eval_every=args.eval_every,
        tb_log_dir=args.tb_log_dir,
    )

    trainer.train()

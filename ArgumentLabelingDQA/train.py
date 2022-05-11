import time
import os
import random
import argparse

import torch
import pickle
import numpy as np
from transformers import DefaultDataCollator, TrainingArguments, Trainer
from transformers import AutoModelForQuestionAnswering

from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange, tqdm
from torch.nn.utils import clip_grad_norm_

from evaluate import *
from dataloader import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_tag", choices=['conll2005', 'conll2009', 'conll2012'])
    # train_path and dev_path can also be cached data directories.
    parser.add_argument("--train_path")
    parser.add_argument("--dev_path")
    parser.add_argument("--pretrained_model_name_or_path")

    # The specific meanings of arg_query_type, argm_query_type and gold_level are provided in the dataloader
    parser.add_argument("--arg_query_type", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--argm_query_type", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gold_level", type=int, choices=[0, 1], default=1)

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_epochs", default=5, type=int)
    parser.add_argument("--warmup_ratio", type=float, default=-1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1)

    parser.add_argument("--resume", action="store_true", help="used to continue training from the checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="checkpoint path when resume is true")

    parser.add_argument("--amp", action="store_true", help="whether to enable mixed precision")
    parser.add_argument("--local_rank", type=int, default=-1)  # DDP has been implemented but has not been tested.
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate during training")
    parser.add_argument("--tensorboard", action='store_true',
                        help="whether to use tensorboard to log training information")
    parser.add_argument("--save", action="store_true", help="whether to save the trained model")
    parser.add_argument("--tqdm_mininterval", default=1, type=float, help="tqdm minimum update interval")
    args = parser.parse_args()
    return args


def train(args, train_data, dev_data, device):
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model_name_or_path).to(device)
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    data_collator = DefaultDataCollator()
    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    # optim = AdamW(model.parameters(), lr=args.lr)
    # for epoch in range(args.max_epochs):
    #     for batch in tqdm(train_loader, f'Epoch {epoch}'):
    #         optim.zero_grad()
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         start_positions = batch['start_positions'].to(device)
    #         end_positions = batch['end_positions'].to(device)
    #         outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
    #                         end_positions=end_positions)
    #         loss = outputs[0]
    #         loss.backward()
    #         optim.step()

    training_args = TrainingArguments(
        output_dir="./dqa_results",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=args.max_epochs,
        weight_decay=0.0001,
        prediction_loss_only=True,
        log_level='info',
        logging_steps=2000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train(True)


if __name__ == "__main__":
    args = args_parser()
    set_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataset = load_data(args.train_path, args.pretrained_model_name_or_path, args.max_tokens,
                              True, args.gold_level, device)
    dev_dataset = load_data(args.dev_path, args.pretrained_model_name_or_path, args.max_tokens, False, args.gold_level,
                            device)
    print(args)
    train(args, train_dataset, dev_dataset, device)

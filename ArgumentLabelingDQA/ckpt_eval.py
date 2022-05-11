import argparse
import pickle
from dataloader import *
from transformers import AutoModelForQuestionAnswering
from evaluate import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='./data/conll2012/dev.english.psense.plabel.conll12.json')
    parser.add_argument("--checkpoint_path", default='deepset/roberta-large-squad2')
    parser.add_argument("--gold_level", type=int, choices=[0, 1], default=1)
    parser.add_argument("--naive_query", action='store_true')
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--amp", action='store_true')
    args = parser.parse_args()

    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint_path)
    evaluation(model, args.data_path, args)

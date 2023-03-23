import json

from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

import os
training_datasets = [
    "./mewnet/wiki_zh_2019/wiki_zh_2019.jsonl",
    "./mewnet/books1/book1.jsonl",
    "./mewnet/aizhigev20/daizhigev20.jsonl",
    "./mewnet/chinese_book/chinese_book.jsonl",
]

total_line = 0
for training_dataset in training_datasets:
    with open(training_dataset, "r", encoding="utf-8") as f:
        for line in f:
            total_line += 1

def training_dataset_iterator():
    for training_dataset in training_datasets:
        with open(training_dataset, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)['text']

# tokenizer.train(training_files, trainer)
tokenizer = old_tokenizer.train_new_from_iterator(training_dataset_iterator(), 52000, total_line)

tokenizer.save_pretrained("checkpoints/tokenizer-shami")

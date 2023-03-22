
import glob
import math
import time
import os
import random
import json
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import tqdm
import random
import gzip
import functools

def write_dataset(path: str, dataset: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

@functools.lru_cache(maxsize=64, typed=False)
def read_dataset(path: str):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def write_gz_dataset(path: str, dataset: List[Dict]):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

@functools.lru_cache(maxsize=64, typed=False)
def read_gz_dataset(path: str):
    dataset = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dir_path: str, model_max_length: int=int(1e30)) -> None:
        self.tokenizer = tokenizer
        self.dir_path = dir_path
        self.files_path = list(glob.glob(os.path.join(self.dir_path, "*.jsonl")))
        self.model_max_length = model_max_length

        self.temp_file = os.path.join(dir_path, "dataset_cache")

        self.data_path = []
        self.chunk_size = 10000
        self.write_cache(self.temp_file)
        self.total_num = (len(self.data_path) - 1) * self.chunk_size + len(read_gz_dataset(self.data_path[-1]))
    
    def write_cache(self, cache_path: str):
        if os.path.exists(cache_path):
            print(f"Cache detected: {cache_path}")
            file_path = os.path.join(self.temp_file, f'*.jsonl.gz')
            self.data_path = sorted(list(glob.glob(file_path)))
            return
        else:
            print(f"Caching dataset: {cache_path}")
            os.makedirs(cache_path)
            chunk_data = []
            chunk_id = 0
            for filepath in self.files_path:
                print(filepath)
                num_lines = sum(1 for line in open(filepath, "r", encoding="utf-8"))
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(tqdm.tqdm(f, total=num_lines)):
                        obj = json.loads(line)
                        if len(obj["text"].strip()) == 0:
                            continue
                        if len(obj["text"]) < 64:
                            continue
                        
                        text = obj["text"]
                        if len(obj["text"]) > 8192:
                            chunks, chunk_size = len(text), 8192
                            text_pieces = [ text[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
                        else:
                            text_pieces = [text]

                        for text_piece in text_pieces:
                            chunk_data.append({"text": text_piece})
                            if len(chunk_data) >= self.chunk_size:
                                file_path = os.path.join(self.temp_file, f'{chunk_id:06d}.jsonl.gz')
                                write_gz_dataset(file_path, chunk_data)
                                self.data_path.append(file_path)
                                chunk_data = []
                                chunk_id += 1

            file_path = os.path.join(self.temp_file, f'{chunk_id:06d}.jsonl.gz')
            write_gz_dataset(file_path, chunk_data)
            self.data_path.append(file_path)
            chunk_data = []
            chunk_id += 1

    def __getitem__(self, index):
        chunk_id = index // self.chunk_size
        chunk = read_gz_dataset(self.data_path[chunk_id])
        chunk_index = index % self.chunk_size
        text = chunk[chunk_index]["text"]
        out = self.tokenizer(text, truncation=True)
        return out

    def __len__(self):
        return self.total_num

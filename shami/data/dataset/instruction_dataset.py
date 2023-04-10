
import glob
import math
import time
import os
import random
import json
from typing import Any, Dict, List, Optional
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import tqdm
import random
import gzip
import functools

CACHE_SIZE = 4

def write_jsonl_dataset(path: str, dataset: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

@functools.lru_cache(maxsize=CACHE_SIZE, typed=False)
def read_jsonl_dataset(path: str):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def write_jsonl_gz_dataset(path: str, dataset: List[Dict]):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

@functools.lru_cache(maxsize=CACHE_SIZE, typed=False)
def read_jsonl_gz_dataset(path: str):
    dataset = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def read_dataset(path: str):
    if path.endswith(".jsonl"):
        return read_jsonl_dataset(path)
    elif path.endswith(".jsonl.gz"):
        return read_jsonl_gz_dataset(path)
    else:
        raise Exception("Unsupported File Extension.")

def write_dataset(path: str, dataset: List[Dict]):
    if path.endswith(".jsonl"):
        write_jsonl_dataset(path, dataset)
    elif path.endswith(".jsonl.gz"):
        write_jsonl_gz_dataset(path, dataset)
    else:
        raise Exception("Unsupported File Extension.")

class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dir_path: str, model_max_length: int=int(1e30), zip: Optional[str]=None) -> None:
        if zip is None:
            self.ext = "jsonl"
        elif zip in ["gz", "gzip"]:
            self.ext = "jsonl.gz"
        self.tokenizer = tokenizer
        self.dir_path = dir_path
        self.files_path = list(glob.glob(os.path.join(self.dir_path, "*.jsonl"), recursive=False))
        self.model_max_length = model_max_length
        self.zip = zip

        self.cache_path = os.path.join(dir_path, "dataset_cache")

        self.data_path = []
        self.chunk_size = 1000
        self.sub_dir_chunk_num = 1000
        self.write_cache(self.cache_path)

   
        last_chunk_item_num = len(read_dataset(self.data_path[-1]))
        self.total_num = (len(self.data_path) - 1) * self.chunk_size + last_chunk_item_num
        random.seed(43)
    
    def write_chunk(self, chunk_data: List[Any], chunk_id: int):
        sub_dir_path = os.path.join(self.cache_path, f'{chunk_id // self.sub_dir_chunk_num:06d}')
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)
        if self.zip is None:
            file_path = os.path.join(sub_dir_path, f'{chunk_id:06d}.jsonl')
        elif self.zip in ["gz", "gzip"]:
            file_path = os.path.join(sub_dir_path, f'{chunk_id:06d}.jsonl.gz')
        else:
            raise Exception("Unsupported zip method.")
        write_dataset(file_path, chunk_data)
        return file_path
    
    def write_cache(self, cache_path: str):
        if os.path.exists(cache_path):
            print(f"Cache detected: {cache_path}")
            file_path = os.path.join(self.cache_path, '*', '*.' + self.ext)
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
                                file_path = self.write_chunk(chunk_data, chunk_id)
                                self.data_path.append(file_path)
                                chunk_data = []
                                chunk_id += 1

            file_path = self.write_chunk(chunk_data, chunk_id)
            self.data_path.append(file_path)
            chunk_data = []
            chunk_id += 1

    def __getitem__(self, index):
        chunk_id = index // self.chunk_size
        chunk = read_dataset(self.data_path[chunk_id])
        chunk_index = index % self.chunk_size
        text = chunk[chunk_index]["text"]
        out = self.tokenizer(text, truncation=True)
        return out

    def __len__(self):
        return self.total_num

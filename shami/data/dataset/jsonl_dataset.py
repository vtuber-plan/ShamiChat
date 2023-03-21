
import math
import time
import os
import random
import json
from typing import Optional
import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import tqdm
import random


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path: str, model_max_length: int=int(1e30)) -> None:
        self.tokenizer = tokenizer
        self.path = path
        self.model_max_length = model_max_length 

        self.data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if len(obj["text"]) < 64:
                    continue
                self.data.append(obj["text"])

    def __getitem__(self, index):
        text = self.data[index]
        out = self.tokenizer(text)
        return out

    def __len__(self):
        return len(self.data)

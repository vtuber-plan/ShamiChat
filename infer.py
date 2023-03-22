import json
import torch
from torch import nn, optim
from torch.nn import functional as F

import glob
import os

files = glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
PATH = sorted(list(files))[-1]
print(f"Loading....{PATH}")

from shami.light_modules.pretrain_shami import PretrainShami
from shami.model.tokenization_shami import ShamiTokenizer
from shami.model.configuration_shami import ShamiConfig

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

config = ShamiConfig.from_json_file("checkpoints/shami-base/config.json")
tokenizer = ShamiTokenizer.from_pretrained("checkpoints/shami-base")
model = PretrainShami.load_from_checkpoint(PATH, config=config)
model.eval()
model = model.to(device)

hparams = model.hparams

inputs = tokenizer(["我是一个人工智能机器人"], return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

# instantiate logits processors
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(10, eos_token_id=model.net.generation_config.eos_token_id),
    ]
)
stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=60)])

outputs = model.net.greedy_search(
    input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
)

out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(out_text)
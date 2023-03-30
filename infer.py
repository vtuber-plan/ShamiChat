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
    BeamSearchScorer,
)


if torch.cuda.is_available():
    device = "cpu"
else:
    device = "cpu"

config = ShamiConfig.from_json_file("checkpoints/shami-1.3B/config.json")
tokenizer = ShamiTokenizer.from_pretrained("checkpoints/shami-1.3B")
model = PretrainShami.load_from_checkpoint(PATH, config=config)
model.eval()
model = model.to(device)

hparams = model.hparams

inputs = tokenizer(["静夜思，李白。"], return_tensors="pt").to(device)
input_ids = inputs["input_ids"]


outputs = model.net.generate(**inputs, penalty_alpha=0.6, top_k=50, max_new_tokens=100)
# outputs = model.net.generate(**inputs, num_beams=5, max_new_tokens=50)
out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(out_text)

# ipmitool -I lanplus -H 114.212.81.248 -U root -P calvin sensor reading "Ambient Temp" "FAN 1 RPM" "FAN 2 RPM" "FAN 3 RPM"
# ipmitool -I lanplus -H 114.212.81.248 -U root -P calvin sdr get "FAN 1 RPM" "FAN 2 RPM" "FAN 3 RPM"
# ipmitool -I lanplus -H 114.212.81.248 -U root -P calvin raw 0x30 0x30 0x01 0x00
# ipmitool -I lanplus -H 114.212.81.248 -U root -P calvin raw 0x30 0x30 0x02 0xff 0x00
# ipmitool -I lanplus -U root -P calvin -H 114.212.81.248 raw 0x30 0x30 0x02 0xff 0x18
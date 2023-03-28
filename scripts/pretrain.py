import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)


import os
import json
import glob
import argparse
import platform
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

from shami.light_modules.pretrain_shami import PretrainShami
from shami.model.modeling_shami import ShamiLayer
from shami.model.configuration_shami import ShamiConfig
from shami.model.tokenization_shami_fast import ShamiTokenizerFast
from shami.model.tokenization_shami import ShamiTokenizer

from shami.data.dataset.pretrain_dataset import PretrainDataset
from shami.hparams import HParams

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
# from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

import lightning_fabric

from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./checkpoints/shami-1.3B/config.json", help='JSON file for configuration')
    parser.add_argument('-p', '--params', type=str, default="./params/shami-1.3B-pretrain.json", help='JSON file for params')
    parser.add_argument('-a', '--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('-d', '--device', type=str, default="0,1,2,3", help='training device ids')
    parser.add_argument('-cp', '--checkpoint', type=str, default="checkpoints/shami-1.3B", help='checkpoint path')
    args = parser.parse_args()

    config = ShamiConfig.from_json_file(args.config)
    hparams = HParams.from_json_file(args.params)

    lightning_fabric.utilities.seed.seed_everything(hparams.seed)

    tokenizer = ShamiTokenizerFast.from_pretrained(args.checkpoint)

    train_dataset = PretrainDataset(tokenizer, "./dataset/pretrain/train", zip="gz")
    valid_dataset = PretrainDataset(tokenizer, "./dataset/pretrain/valid", zip="gz")
        
    collate_fn = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=4, shuffle=True, pin_memory=True, collate_fn=collate_fn, prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=hparams.valid_batch_size, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model = PretrainShami(config, hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=None, save_last=True, every_n_train_steps=1, save_weights_only=False,
    )
    # monitor="val_loss", mode="min", save_top_k=5
    # earlystop_callback = EarlyStopping(monitor="valid/loss_mel_epoch", mode="min", patience=13)

    devices = [int(n.strip()) for n in args.device.split(",")]
    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback],
    }

    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if platform.system().lower() == 'windows':
        backend = "gloo"
    else:
        backend = "nccl"
    if "strategy" in hparams:
        if hparams.strategy == "fsdp":
            from pytorch_lightning.strategies import FSDPStrategy
            fsdp = FSDPStrategy(
                activation_checkpointing=ShamiLayer,  # or pass a list with multiple types
                process_group_backend=backend
            )
            trainer_params["strategy"] = fsdp
        elif hparams.strategy == "ddp":
            ddp = DDPStrategy(process_group_backend=backend, find_unused_parameters=True)
            trainer_params["strategy"] = ddp
    elif len(devices) > 1:
        ddp = DDPStrategy(process_group_backend=backend, find_unused_parameters=True)
        trainer_params["strategy"] = ddp

    if hparams.fp16:
        print("using fp16")
        trainer_params["precision"] = "16-mixed"
    elif hparams.bf16:
        print("using bf16")
        trainer_params["precision"] = "bf16-mixed"
    
    if "max_epochs" in hparams:
        trainer_params["max_epochs"] = hparams["max_epochs"]
    if "accumulate_grad_batches" in hparams:
        trainer_params["accumulate_grad_batches"] = hparams["accumulate_grad_batches"]
    # profiler = AdvancedProfiler(filename="profile.txt")
    
    trainer = pl.Trainer(**trainer_params) # , profiler=profiler, max_steps=200
    # resume training
    ckpt_path = None
    if os.path.exists("lightning_logs"):
        versions = glob.glob("lightning_logs/version_*")
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
  main()
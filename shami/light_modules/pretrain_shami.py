


from typing import Dict
import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl

from .. import hparams
from ..model.configuration_shami import ShamiConfig
from ..model.modeling_shami import ShamiModel, ShamiLMHeadModel

class PretrainShami(pl.LightningModule):
    def __init__(self, config: ShamiConfig, params: hparams) -> None:
        super().__init__()
        self.config = config
        self.params = params

        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        self.net = ShamiLMHeadModel(config)
        self.save_hyperparameters(ignore=["config"])

    def forward(self, tokens: Dict[str, torch.Tensor]):
        input_ids, input_mask = tokens["input_ids"], tokens["attention_mask"]
        batch_size = input_ids.shape[0]

        lm_logits = self.net(
            input_ids=input_ids,
            attention_mask=input_mask,
            use_cache=False,
        )
        return lm_logits

    def training_step(self, batch, batch_idx: int):
        input_ids, input_mask = batch["input_ids"], batch["attention_mask"]

        batch_size = input_ids.shape[0]
        source_tokens = {
            'input_ids': input_ids[..., :-1],
            'attention_mask': input_mask[..., :-1]
        }

        lm_logits = self.forward(
            tokens=source_tokens,
        )

        shift_label_ids = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.vocab_size), shift_label_ids.view(-1))

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask = batch["input_ids"], batch["attention_mask"]

        batch_size = input_ids.shape[0]
        source_tokens = {
            'input_ids': input_ids[..., :-1],
            'attention_mask': input_mask[..., :-1]
        }

        lm_logits = self.forward(
            tokens=source_tokens
        )

        shift_label_ids = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(lm_logits.view(-1, self.vocab_size), shift_label_ids.view(-1))
        self.log('val_loss', loss)

    
    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.net.parameters(), 
            self.hparams.params.train.learning_rate, 
            betas=self.hparams.params.train.betas, 
            eps=self.hparams.params.train.eps)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.hparams.params.train.lr_decay)
        self.scheduler.last_epoch = self.current_epoch - 1

        return [self.optim], [self.scheduler]

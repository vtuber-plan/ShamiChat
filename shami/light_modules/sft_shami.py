


from typing import Dict
import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl

from .. import hparams
from ..model.configuration_shami import ShamiConfig
from ..model.modeling_shami import ShamiModel, ShamiLMHeadModel

class SupervisedFineTuningShami(pl.LightningModule):
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
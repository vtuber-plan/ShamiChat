"""
 PyTorch Shami model.
"""
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    CausalLMOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from .configuration_shami import ShamiConfig


logger = logging.get_logger(__name__)


SHAMI_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shami-base",
    "shami-large",
]

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# RoPE旋转位置编码
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@dataclass
class ShamiModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ShamiAttention(nn.Module):
    def __init__(self, config: ShamiConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        # self.cache_k = torch.zeros(
        #     (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
        # ).cuda()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

class ShamiFeedForward(nn.Module):
    def __init__(self, config: ShamiConfig):
        super().__init__()
        self.d_model: int = config.d_model
        self.hidden_dim: int = 4 * config.d_model
        self.multiple_of: int = config.multiple_of

        self.ffn_hidden_dim = int(2 * self.hidden_dim / 3)
        self.ffn_hidden_dim = self.multiple_of * ((self.ffn_hidden_dim + self.multiple_of - 1) // self.multiple_of)

        self.w1 = nn.Linear(self.d_model, self.ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_hidden_dim, self.d_model, bias=False)
        self.w3 = nn.Linear(self.d_model, self.ffn_hidden_dim, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ShamiLayer(nn.Module):
    def __init__(self, layer_id: int, config: ShamiConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.attention = ShamiAttention(config)
        self.feed_forward = ShamiFeedForward(config)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class ShamiPreTrainedModel(PreTrainedModel):
    config_class = ShamiConfig
    base_model_prefix = "transformer"
    # is_parallelizable = True
    # supports_gradient_checkpointing = True
    _no_split_modules = ["ShamiLayer"]

    def __init__(self, config: ShamiConfig):
        super().__init__(config)


class ShamiModel(ShamiPreTrainedModel):
    def __init__(self, config: ShamiConfig):
        super().__init__(config)

        self.d_model: int = config.d_model
        self.n_layers: int = config.n_layers
        self.n_heads: int = config.n_heads
        self.vocab_size: int = config.vocab_size  # defined later by tokenizer
        self.multiple_of: int = config.multiple_of  # make SwiGLU hidden layer size multiple of large power of 2
        self.norm_eps: float = config.norm_eps
        self.max_seq_len: int = config.max_seq_len

        self.word_embedding = nn.Embedding(self.vocab_size, self.d_model)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(ShamiLayer(layer_id, config))

        self.norm = RMSNorm(self.d_model, eps=self.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.d_model // self.n_heads, self.max_seq_len * 2
        )
        self.freqs_cis_device_mapping = {}

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # delete after depreciation warning is removed
    ) -> Union[Tuple, ShamiModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.word_embedding(input_ids)
        hidden_states = inputs_embeds

        batch_size, seqlen = input_ids.shape
        start_pos = 0
        if hidden_states.device not in self.freqs_cis_device_mapping:
            self.freqs_cis_device_mapping[hidden_states.device] = self.freqs_cis.to(hidden_states.device)
        freqs_cis = self.freqs_cis_device_mapping[hidden_states.device]
        freqs_cis = freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=inputs_embeds.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(hidden_states)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states, freqs_cis, mask)
        h = self.norm(hidden_states)
        return ShamiModelOutput(
            last_hidden_state=h,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )


class ShamiLMHeadModel(ShamiPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_loss.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.lm_model = ShamiModel(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs
        ) -> dict:

        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ShamiModelOutput]:
        transformer_outputs = self.lm_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # (batch_size, sequence_length, hidden_size)
        hidden_states = transformer_outputs.last_hidden_state
        # (batch_size, sequence_length, vocab_size)
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )
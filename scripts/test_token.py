import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from shami.model.tokenization_shami import ShamiTokenizer
from shami.model.tokenization_shami_fast import  ShamiTokenizerFast

fast_tokenizer = ShamiTokenizerFast.from_pretrained("checkpoints/shami-base")
out_tokens = fast_tokenizer.tokenize("谢谢你的回复！GitHub账号是akemimadoka。")
out_ids = fast_tokenizer.convert_tokens_to_ids(out_tokens)
out_str = fast_tokenizer.convert_tokens_to_string(out_tokens)
print(out_tokens)
print(out_ids)
print(out_str)

tokenizer = ShamiTokenizer.from_pretrained("checkpoints/shami-base")
out_tokens = tokenizer.tokenize("谢谢你的回复！GitHub账号是akemimadoka。")
out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
out_str = tokenizer.convert_tokens_to_string(out_tokens)
print(out_tokens)
print(out_ids)
print(out_str)


out_tokens = tokenizer("谢谢你的回复！GitHub账号是akemimadoka。")
print(out_tokens)

out_tokens = fast_tokenizer("谢谢你的回复！GitHub账号是akemimadoka。")
print(out_tokens)
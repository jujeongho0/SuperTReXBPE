import os
import datasets
from pathlib import Path
from transformers import AutoTokenizer

def bytes_per_token(texts, tokenizer):
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        total_bytes += len(text["text"].encode("utf-8"))
        total_tokens += len(tokenizer.encode(text["text"], add_special_tokens=False))

    return total_bytes / total_tokens

data_dir = "/workspace/data/NetApp/ISTD_VOL01/lm_team/personal/jeongho/valid"

TOKENIZER_PATH = "./trex_30gb_128k"

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)
except:
    TOKENIZER_PATH = Path(TOKENIZER_PATH) / "n1"
    pp = TOKENIZER_PATH / "post_processed"
    tokenizer = AutoTokenizer.from_pretrained(pp, use_fast=True, trust_remote_code=True) if pp.exists() \
        else AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)

print("### Bytes-per-Token")
for col in os.listdir(data_dir):
    texts = datasets.load_from_disk(os.path.join(data_dir, col))
    bpt = bytes_per_token(texts, tokenizer)
    print(f"{col}: {bpt}")
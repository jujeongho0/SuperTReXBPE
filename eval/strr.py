import os
from pathlib import Path
from transformers import AutoTokenizer

data_dir = "./data"
language_files = {
    "korean": "korean_words.txt",
    "hanja": "hanja_words.txt",
    "english": "english_words.txt",
    "chinese": "chinese_words.txt",
    "japanese": "japanese_words.txt",
}

TOKENIZER_PATH = "./trex_30gb_128k"

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)
except:
    TOKENIZER_PATH = Path(TOKENIZER_PATH) / "n1"
    pp = TOKENIZER_PATH / "post_processed"
    tokenizer = AutoTokenizer.from_pretrained(pp, use_fast=True, trust_remote_code=True) if pp.exists() \
        else AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, trust_remote_code=True)

print("### STRR")
for k, v in language_files.items():
    entries = []
    with open(os.path.join(data_dir, v), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
    
            entries.append(line)

    single_token_count_lang = sum(1 for entry in entries if len(tokenizer.encode(entry, add_special_tokens=False)) == 1)
    strr_lang = (single_token_count_lang / len(entries)) * 100 if entries else 0

    print(f"{k.capitalize()}: {strr_lang:.2f}%")
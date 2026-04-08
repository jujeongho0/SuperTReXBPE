import os
import re
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers, Regex, normalizers, processors, decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE

from mecab import MeCab

_HANGUL_RE = re.compile(r"[가-힣]+")
mecab = MeCab()

JOSA_TOKEN = "▁"

def tokenize_korean(text):
    pieces = []
    last = 0

    for m in _HANGUL_RE.finditer(text):
        start, end = m.span()
        if start > last:
            pieces.append(text[last:start])

        korean_chunk = text[start:end]
        tokens = []
        morphs = mecab.pos(korean_chunk)

        buffer = ""
        for morph, pos in morphs:
            if pos.startswith("J"):
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(morph)
            else:
                buffer += morph

        if buffer:
            tokens.append(buffer)

        pieces.append(JOSA_TOKEN.join(tokens))
        last = end

    if last < len(text):
        pieces.append(text[last:])

    return "".join(pieces)

def tokenized_corpus(text_files):
    for path in text_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                yield tokenize_korean(text)

def build_special_tokens():
    special_tokens = [
        "<|startoftext|>", # bos_token
        "<|return|>", # eos_token
        "<|endoftext|>", # pad_token
    ]

    special_tokens += [" " * i for i in range(2, 30)]
    special_tokens += ["\n" * i for i in range(2, 10)]
    special_tokens += ["\t" * i for i in range(2, 10)]
    
    return special_tokens


SPECIAL_TOKENS = build_special_tokens()

def train_tokenizer(text_files: list[str], vocab_size: int = 100000, regex_string: str = None):
    tokenizer = Tokenizer(BPE())

    tokenizer.normalizer = normalizers.NFC()
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.Split(
            pattern=Regex(JOSA_TOKEN),
            behavior="removed",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=False,
            use_regex=False,
        ),
    ])

    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=True, trim_offsets=False, use_regex=True)
    
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    iterator = tokenized_corpus(text_files)
    tokenizer.train_from_iterator(iterator, trainer)

    return tokenizer

def extend_tokenizer(text_files: list[str], vocab_size: int = 100000, regex_string: str = None):
    tokenizer = Tokenizer(BPE())

    tokenizer.normalizer = normalizers.NFC()
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=False,
            use_regex=False,
        ),
    ])
    
    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=True, trim_offsets=False, use_regex=True)

    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(text_files, trainer)

    return tokenizer

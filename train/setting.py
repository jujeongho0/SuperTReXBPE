import os
import re
from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers, Regex, normalizers, processors, decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE

from mecab import MeCab

BASE_DIR = Path(__file__).resolve().parent

_HANGUL_RE = re.compile(r"[가-힣]+")
mecab = MeCab()

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

        pieces.append("▁".join(tokens))
        last = end

    if last < len(text):
        pieces.append(text[last:])

    return "".join(pieces)

def tokenized_corpus(text_files):
    for path in text_files:
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    texts.append("")
                    continue

                texts.append(tokenize_korean(text))
        
        yield "\n".join(texts).strip()

def build_special_tokens():
    special_tokens = [
        "<unk>", # unk_token
        "<|startoftext|>", # bos_token
        "<|return|>", # eos_token
        "<|endoftext|>", # pad_token
    ]

    special_tokens += [" " * i for i in range(31, 1, -1)]
    special_tokens += ["\t" * i for i in range(9, 1, -1)]
    special_tokens += ["\n" * i for i in range(9, 1, -1)]
    
    return special_tokens

def build_initial_alphabet():
    alphabet = set()

    alphabet.update(chr(c) for c in range(0xAC00, 0xD7A4)) # Hangul Syllables
    alphabet.update(chr(c) for c in range(0x3131, 0x3164)) # Hangul Compatibility Jamo

    # Hanja
    with open(BASE_DIR / "../data/hanja_level1.txt", "r", encoding="utf-8") as f:
        alphabet.update(line.strip() for line in f if line.strip())

    # Kanji
    with open(BASE_DIR / "../data/kanji.txt", "r", encoding="utf-8") as f:
        alphabet.update(line.strip() for line in f if line.strip())

    # Hiragana
    alphabet.update(chr(c) for c in range(0x3041, 0x3094))

    # Katakana
    alphabet.update(chr(c) for c in range(0x30A1, 0x30FA))

    return sorted(alphabet)

SPECIAL_TOKENS = build_special_tokens()
INITIAL_ALPHABET = build_initial_alphabet()
HEX_TOKENS = [f"<0x{i:02X}>" for i in range(256)]
LIMIT_ALPHABET = 1000 + len(INITIAL_ALPHABET)

def train_tokenizer(text_files: list[str], vocab_size: int = 100000, regex_string: str = None):
    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
            fuse_unk=True,
            byte_fallback=True,
        )
    )

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(), 
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
        # FIXME: For tokenize_korean()
        pre_tokenizers.Split(
            pattern=Regex("▁"),
            behavior="removed",
            invert=False,
        ),
    ])
    
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[*SPECIAL_TOKENS, *HEX_TOKENS],
        limit_alphabet=LIMIT_ALPHABET,
        initial_alphabet=INITIAL_ALPHABET,
        show_progress=True,
    )

    iterator = tokenized_corpus(text_files)
    tokenizer.train_from_iterator(iterator, trainer)

    return tokenizer

def extend_tokenizer(text_files: list[str], vocab_size: int = 100000, regex_string: str = None):
    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>",
            fuse_unk=True,
            byte_fallback=True,
        )
    )

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(), 
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
    ])
    
    tokenizer.decoder = decoders.Sequence([
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[*SPECIAL_TOKENS, *HEX_TOKENS],
        limit_alphabet=LIMIT_ALPHABET,
        initial_alphabet=INITIAL_ALPHABET,
        show_progress=True,
    )

    tokenizer.train(text_files, trainer)

    return tokenizer

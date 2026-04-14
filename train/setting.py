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

SPACE_TOKEN = "▁"
JOSA_TOKEN = "▂"

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

def build_initial_alphabet():
    alphabet = set()

    # alphabet.update(chr(c) for c in range(0xAC00, 0xD7A4)) # Hangul Syllables
    # alphabet.update(chr(c) for c in range(0x3131, 0x3164)) # Hangul Compatibility Jamo

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

INITIAL_ALPHABET = build_initial_alphabet()

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
        normalizers.Replace(" ", SPACE_TOKEN),
    ])
    
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
    ])
    
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace(SPACE_TOKEN, " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])

    added_tokens = []
    added_tokens += [SPACE_TOKEN * i for i in range(2, 30)]
    added_tokens += ["\n" * i for i in range(2, 10)]
    added_tokens += ["\t" * i for i in range(2, 10)]
    added_tokens += INITIAL_ALPHABET
    
    for at in added_tokens:
        tokenizer.add_tokens(
            AddedToken(
                content=at,
                rstrip=False,
                lstrip=False,
                single_word=False,
                normalized=False,
                special=False
            )
        )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<|startoftext|>", "<|return|>", "<|endoftext|>"] + [f"<0x{i:02X}>" for i in range(256)],
        limit_alphabet=1000,
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
        normalizers.Replace(" ", SPACE_TOKEN),
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
    ])
    
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace(SPACE_TOKEN, " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])

    added_tokens = []
    added_tokens += [SPACE_TOKEN * i for i in range(2, 30)]
    added_tokens += ["\n" * i for i in range(2, 10)]
    added_tokens += ["\t" * i for i in range(2, 10)]
    added_tokens += INITIAL_ALPHABET
    
    for at in added_tokens:
        tokenizer.add_tokens(
            AddedToken(
                content=at,
                rstrip=False,
                lstrip=False,
                single_word=False,
                normalized=False,
                special=False
            )
        )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<|startoftext|>", "<|return|>", "<|endoftext|>"] + [f"<0x{i:02X}>" for i in range(256)],
        limit_alphabet=1000,
        show_progress=True,
    )

    tokenizer.train(text_files, trainer)

    return tokenizer

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
        morphs = mecab.morphs(korean_chunk)
        pieces.append("▁".join(morphs))

        last = end

    if last < len(text):
        pieces.append(text[last:])

    return "".join(pieces)

def tokenized_corpus(text_files):
    for path in text_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield tokenize_korean(line)
        except:
            print(path)
            exit()

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

    alphabet.update(chr(c) for c in range(0x3041, 0x3097)) # Hiragana
    alphabet.update(chr(c) for c in range(0x30A1, 0x30FB)) # Katakana

    return sorted(alphabet)

def train_tokenizer(text_files: list[str], vocab_size: int = 100000, regex_string: str = None):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        show_progress=True,
        initial_alphabet=build_initial_alphabet(), # FIXME
        vocab_size=vocab_size,
    )

    tokenizer.normalizer = normalizers.NFC()
    
    pretokenizers = [
        pre_tokenizers.Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
        # FIXME
        pre_tokenizers.Split(
            pattern=Regex("▁"),
            behavior="removed",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=True,
            use_regex=False,
        ),
    ]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pretokenizers)
    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=True, trim_offsets=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)

    # FIXME
    iterator = tokenized_corpus(text_files)
    tokenizer.train_from_iterator(iterator, trainer)

    return tokenizer

def extend_tokenizer(text_files: list[str], vocab_size: int = 100000, regex_string: str = None):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        show_progress=True,
        initial_alphabet=build_initial_alphabet(), # FIXME
        vocab_size=vocab_size,
    )

    tokenizer.normalizer = normalizers.NFC()
    
    pretokenizers = [
        pre_tokenizers.Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=True,
            use_regex=False,
        ),
    ]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pretokenizers)
    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=True, trim_offsets=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)

    tokenizer.train(text_files, trainer)

    return tokenizer
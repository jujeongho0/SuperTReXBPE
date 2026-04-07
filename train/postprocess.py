from transformers import PreTrainedTokenizerFast
from tokenizers import AddedToken
import click
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


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


INITIAL_ALPHABET = build_initial_alphabet()


@click.command()
@click.option(
    "--process_tgt",
    type=str,
)
@click.option(
    "--output_dir",
    type=str,
)
def main(
    process_tgt: str,
    output_dir: str,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    new_tok = PreTrainedTokenizerFast.from_pretrained(process_tgt)

    # FIXME
    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131054|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131055|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|constrain|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131057|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|channel|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|start|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|end|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )
    
    new_tok.add_tokens(
        AddedToken(
            content="<|message|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131062|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131063|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131064|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|call|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131066|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131067|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131068|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131069|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|reserved_131070|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.add_tokens(
        AddedToken(
            content="<|endofprompt|>",
            rstrip=False,
            lstrip=False,
            single_word=False,
            normalized=False,
            special=True
        )
    )

    new_tok.bos_token = "<|startoftext|>"
    new_tok.eos_token = "<|return|>"
    new_tok.pad_token = "<|endoftext|>"
    
    new_tok.save_pretrained(output_dir)

    new_tok.model_input_names = ['input_ids', 'attention_mask']
    
    with open(os.path.join(output_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    update_data = {
        "model_input_names": [
            "input_ids",
            "attention_mask"
        ],
    }

    config.update(update_data)

    new_decoder = {}
    for k, v in config["added_tokens_decoder"].items():
        content = v.get("content")
        if content in INITIAL_ALPHABET:
            continue
        new_decoder[k] = v

    config["added_tokens_decoder"] = new_decoder

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    new_added_tokens = []
    for item in tokenizer_json["added_tokens"]:
        content = item.get("content")
        if content in INITIAL_ALPHABET:
            continue
        new_added_tokens.append(item)

    tokenizer_json["added_tokens"] = new_added_tokens

    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

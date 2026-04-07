from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPACE_TOKEN = "▁"
JOSA_TOKEN = "▂"

HEX_TOKENS = {f"<0x{i:02X}>" for i in range(256)}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fix_tokenizer_config(tokenizer_config: dict[str, Any]) -> dict[str, Any]:
    decoder = tokenizer_config.get("added_tokens_decoder")

    new_decoder: dict[str, Any] = {}
    for k, v in decoder.items():
        content = v.get("content")
        if content in HEX_TOKENS:
            continue
        new_decoder[k] = v

    tokenizer_config["added_tokens_decoder"] = new_decoder
    return tokenizer_config


def fix_added_tokens(tokenizer_json: dict[str, Any]) -> dict[str, Any]:
    added_tokens = tokenizer_json.get("added_tokens")

    new_added_tokens = []
    for item in added_tokens:
        content = item.get("content")
        if content in HEX_TOKENS:
            continue
        new_added_tokens.append(item)

    tokenizer_json["added_tokens"] = new_added_tokens
    return tokenizer_json


def fix_pre_tokenizer(node: Any) -> Any:
    new_list = []
    for item in node["pretokenizers"]:
        typ = item.get("type")

        if typ == "Split":
            pattern = item.get("pattern")
            pattern_regex = pattern.get("Regex")

            if pattern_regex == JOSA_TOKEN:
                continue

            new_list.append(item)

    node["pretokenizers"] = new_list  

    return node


def fix_tokenizer_json(tokenizer_json: dict[str, Any]) -> dict[str, Any]:
    tokenizer_json = fix_added_tokens(tokenizer_json)
    tokenizer_json["pre_tokenizer"] = fix_pre_tokenizer(tokenizer_json["pre_tokenizer"])

    return tokenizer_json


def convert_tokenizer_files(src_dir: str | Path, dst_dir: str | Path) -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    special_tokens_map_path = src_dir / "special_tokens_map.json"
    tokenizer_config_path = src_dir / "tokenizer_config.json"
    tokenizer_json_path = src_dir / "tokenizer.json"

    special_tokens_map = load_json(special_tokens_map_path)
    save_json(dst_dir / "special_tokens_map.json", special_tokens_map)

    tokenizer_config = load_json(tokenizer_config_path)
    tokenizer_config = fix_tokenizer_config(tokenizer_config)

    save_json(dst_dir / "tokenizer_config.json", tokenizer_config)

    tokenizer_json = load_json(tokenizer_json_path)
    tokenizer_json = fix_tokenizer_json(tokenizer_json)
    save_json(dst_dir / "tokenizer.json", tokenizer_json)


if __name__ == "__main__":
    TOKENIZER_NAME = "./supertrexbpe_30gb_88k_extend_128k"
    SRC_DIR = f"{TOKENIZER_NAME}/n1/post_processed"
    DST_DIR = f"{TOKENIZER_NAME}/n1/converted"

    convert_tokenizer_files(
        src_dir=SRC_DIR,
        dst_dir=DST_DIR,
    )

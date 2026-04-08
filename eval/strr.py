import os
import json
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


def load_word_list(file_path):
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(line)
    return entries


def compute_strr(entries, tokenizer):
    if not entries:
        return None

    single_token_count = sum(
        1
        for entry in entries
        if len(tokenizer.encode(entry, add_special_tokens=False)) == 1
        # if len(tokenizer.encode(entry, add_special_tokens=False)) == 1 or len(tokenizer.encode(f" {entry}", add_special_tokens=False)) == 1
    )

    return (single_token_count / len(entries)) * 100


def load_tokenizers(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_tokenizers(language_files, data_dir, tokenizer_config):
    results = {
        "summary": {},
        "per_model": {},
        "per_language": {},
    }

    language_names = sorted(language_files.keys())
    for lang in language_names:
        results["per_language"][lang] = {}

    print("## STRR")

    for model_name, model_info in tokenizer_config.items():
        tokenizer_path = model_info["path"]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
        )

        print(f"\n### Model: {model_name}")
        results["per_model"][model_name] = {}

        model_scores = []

        for lang in language_names:
            file_name = language_files[lang]
            file_path = os.path.join(data_dir, file_name)

            entries = load_word_list(file_path)
            strr = compute_strr(entries, tokenizer)

            results["per_model"][model_name][lang] = strr
            results["per_language"][lang][model_name] = strr

            if strr is not None:
                model_scores.append(strr)
                print(f"{lang.capitalize()}: {strr:.2f}%")
            else:
                print(f"{lang.capitalize()}: None")

        avg_strr = sum(model_scores) / len(model_scores) if model_scores else None
        results["summary"][model_name] = {
            "avg_strr": avg_strr
        }

        if avg_strr is not None:
            print(f"Average: {avg_strr:.2f}%")
        else:
            print("Average: None")

    return results


def save_results_json(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    tokenizer_config_path = "./eval/baseline_tokenizers.json"
    output_json_path = "./eval/results/tokenizer_strr_results.json"

    tokenizer_config = load_tokenizers(tokenizer_config_path)
    results = evaluate_tokenizers(language_files, data_dir, tokenizer_config)

    save_results_json(results, output_json_path)
    print(f"\nSaved JSON: {output_json_path}")

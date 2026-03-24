import os
import json
import datasets
from pathlib import Path
from transformers import AutoTokenizer


def bytes_per_token(texts, tokenizer):
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        content = text["text"]
        total_bytes += len(content.encode("utf-8"))
        total_tokens += len(tokenizer.encode(content, add_special_tokens=False))

    if total_tokens == 0:
        return None

    return total_bytes / total_tokens


def load_tokenizers(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_tokenizers(data_dir, tokenizer_config):
    results = {
        "summary": {},
        "per_model": {},
        "per_dataset": {}
    }

    data_dir = Path(data_dir)
    dataset_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])

    for dataset_name in dataset_names:
        results["per_dataset"][dataset_name] = {}

    print("## Bytes-per-Token")

    for model_name, model_info in tokenizer_config.items():
        tokenizer_path = model_info["path"]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True
        )

        print(f"\n### Model: {model_name}")
        results["per_model"][model_name] = {}

        model_scores = []

        for dataset_name in dataset_names:
            dataset_path = data_dir / dataset_name
            texts = datasets.load_from_disk(str(dataset_path))
            bpt = bytes_per_token(texts, tokenizer)

            results["per_model"][model_name][dataset_name] = bpt
            results["per_dataset"][dataset_name][model_name] = bpt

            if bpt is not None:
                model_scores.append(bpt)
                print(f"{dataset_name}: {bpt:.6f}")
            else:
                print(f"{dataset_name}: None")

        avg_bpt = sum(model_scores) / len(model_scores) if model_scores else None
        results["summary"][model_name] = {
            "avg_bytes_per_token": avg_bpt
        }

        if avg_bpt is not None:
            print(f"Average: {avg_bpt:.6f}")
        else:
            print("Average: None")

    return results


def save_results_json(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    data_dir = "/path/to/valid"
    tokenizer_config_path = "./eval/baseline_tokenizers.json"
    output_json_path = "./eval/results/tokenizer_bpt_results.json"

    tokenizer_config = load_tokenizers(tokenizer_config_path)
    results = evaluate_tokenizers(data_dir, tokenizer_config)

    save_results_json(results, output_json_path)
    print(f"\nSaved JSON: {output_json_path}")

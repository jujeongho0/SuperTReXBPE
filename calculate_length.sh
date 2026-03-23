#!/bin/bash
set -euo pipefail

IFS=' ' read -r -a TOKENIZER_PATHS <<< "${TREX_TOKENIZER_PATHS:-./k-exaone-236b-a23b ./trex_1gb_64k}"
DATA_ROOT="${TREX_VALID_DATA_ROOT:-/workspace/data/NetApp/ISTD_VOL01/lm_team/personal/jeongho/valid}"
OUTPUT_ROOT="${TREX_RECORD_ROOT:-./}"

mkdir -p "$OUTPUT_ROOT"

for TOK in "${TOKENIZER_PATHS[@]}"; do
    echo ">>> Running for $TOK"

    TOK_NAME=$(basename "$TOK")
    OUT_PKL="${OUTPUT_ROOT}/${TOK_NAME}_lang_results.pkl"

    python3 -m train.calculate_length \
        --ds_root "$DATA_ROOT" \
        --tok_root "$TOK" \
        --out_pkl "$OUT_PKL"

    if [ -d "$DATA_ROOT" ]; then
        find "$DATA_ROOT" -maxdepth 2 -name "cache*" -type d -exec rm -rf {} +
    fi
done
#!/bin/bash
set -euo pipefail

DATA_ROOT="${TREX_DATA_ROOT:-/workspace/data/NetApp/ISTD_VOL01/lm_team/personal/jeongho/train}"

run_train () {
    CONFIG_DIR="$1"
    DATA_BYTES="$2"
    VOCAB_SIZE="$3"
    OUT_NAME="$4"

    if [ ! -d "$CONFIG_DIR" ]; then
        echo "❌ Config directory not found: $CONFIG_DIR" >&2
        exit 1
    fi

    for CFG_FILE in "$CONFIG_DIR"/*.yaml; do
        [ -e "$CFG_FILE" ] || continue
        BASENAME=$(basename "$CFG_FILE" .yaml)

        echo "🚀 Starting training with config: ${CFG_FILE}"

        OUTPUT_DIR="${OUT_NAME}/${BASENAME}"

        if [ -d "$OUTPUT_DIR" ]; then
            echo "⚠️ ${OUTPUT_DIR} already exists. Skipping train/preprocess."
            continue
        fi

        python3 -m train.train \
            --data_root "$DATA_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --cfg_file "$CFG_FILE" \
            --num_bytes "$DATA_BYTES" \
            --vocab_size "$VOCAB_SIZE" \
            --regex_string "[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

        python3 -m train.preprocess \
            --process_tgt "$OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR/post_processed"

        sleep 3
    done
}

run_train config_mn 1073741824 65515 trex_1gb_64k
run_train config_mn_optimal 32212254720 131051 trex_30gb_128k
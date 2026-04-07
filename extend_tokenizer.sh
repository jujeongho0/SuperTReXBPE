#!/bin/bash
set -euo pipefail

DATA_ROOT="${TREX_DATA_ROOT:-/path/to/train}"

run_train () {
    CONFIG_DIR="$1"
    ORIG_TOKENIZER_NAME="$2"
    NUM_INHERIT_MERGES="$3"
    VOCAB_SIZE="$4"
    OUT_NAME="$5"

    if [ ! -d "$CONFIG_DIR" ]; then
        echo "❌ Config directory not found: $CONFIG_DIR" >&2
        exit 1
    fi

    for CFG_FILE in "$CONFIG_DIR"/*.yaml; do
        [ -e "$CFG_FILE" ] || continue
        BASENAME=$(basename "$CFG_FILE" .yaml)

        echo "🚀 Starting training with config: ${CFG_FILE}"

        ORIG_TOKENIZER_DIR="${ORIG_TOKENIZER_NAME}/${BASENAME}"

        if [ ! -d "$ORIG_TOKENIZER_DIR" ]; then
            echo "❌ ${ORIG_TOKENIZER_DIR} not found: train/preprocess."
            continue
        fi

        OUTPUT_DIR="${OUT_NAME}/${BASENAME}"

        if [ -d "$OUTPUT_DIR" ]; then
            echo "⚠️ ${OUTPUT_DIR} already exists. Skipping train/preprocess."
            continue
        fi

        mkdir -p $OUTPUT_DIR
        head -n $((NUM_INHERIT_MERGES + 1)) $ORIG_TOKENIZER_DIR/merges.txt > $OUTPUT_DIR/merges.txt
        # this line copies over the training data. if you wish to use a different training corpus, delete this line and provide --num_bytes.
        cp $ORIG_TOKENIZER_DIR/meta.json $OUTPUT_DIR/meta.json

        python3 -m train.extend \
            --data_root "$DATA_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --cfg_file "$CFG_FILE" \
            --vocab_size "$VOCAB_SIZE" \
            --regex_string "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?:▁\p{L}\p{M}*)*)+|\p{N}{1,3}|▁?[^▁\s\p{L}\p{N}]+[\r\n]?|[▁\s]*[\r\n]|[▁\s]+(?![^▁\s])|[▁\s]+"

        python3 -m train.postprocess \
            --process_tgt "$OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR/post_processed"

        sleep 3
    done
}

run_train config_mn_optimal trex_30gb_128k 90112 131054 supertrexbpe_30gb_88k_extend_128k

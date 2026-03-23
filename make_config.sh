#!/bin/bash
set -euo pipefail

OUTPUT_FOLDER="${TREX_CONFIG_OUTPUT_DIR:-./config_mn}"
NUM_CONFIGS="${TREX_NUM_CONFIGS:-512}"

python3 -m mixture.make_config \
    --output_folder "${OUTPUT_FOLDER}" \
    --num_configs "${NUM_CONFIGS}"
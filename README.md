# SuperTReXBPE

This repository combines [TReX](https://github.com/HanGyeol-Yoo/TReX) and [SuperBPE](https://github.com/PythonNut/superbpe) into a single pipeline for building optimized tokenizers for language model pretraining.

It first provides utilities and experiments for tokenizer design and data mixture optimization in the spirit of TReX. This includes:

- Scripts for training and post-processing multiple tokenizers.
- Tools for generating dataset mixtures via Dirichlet sampling.
- A normalized-sequence-length analysis pipeline to measure tokenization efficiency.
- Research workflows for fitting regression models that identify optimal data mixtures.

Based on the optimized mixture and tokenizer insights obtained from this stage, the repository then applies the SuperBPE methodology to train a final tokenizer. It includes:

- Instructions and configurations for training SuperBPE tokenizers.
- Analysis tools for evaluating tokenizer performance. (e.g. bytes-per-token, single-token-retention-rate)

In summary, this repository enables a two-stage approach:

> **TReX is used to explore and determine the optimal data mixture and tokenizer characteristics, and SuperBPE is used to train the final tokenizer tailored to that optimized setting.**

This design allows users to move beyond off-the-shelf tokenizers and build custom, data-aware tokenizers suited to their specific training objectives.

## Setup
First, clone the project with:
```bash
git clone --recurse-submodules https://github.com/jujeongho0/superTReXbpe.git
cd superTReXbpe
pip install -r requirements.txt
```
We use a custom [fork](https://github.com/alisawuffles/tokenizers-superbpe) of [huggingface/tokenizers](https://github.com/huggingface/tokenizers) which conflicts with the original.
Because of this, we recommend *always installing this project in its own virtual environment.*

## Generating Mixture Configurations

Use `mixture/make_config.py` to sample domain weightings for regression experiments. The script implements a temperature-controlled Dirichlet sampling strategy with clipping bounds derived from empirical token usage. Generate 512 candidate mixtures into `config_mn/` with:

```bash
python3 -m mixture.make_config --output_folder config_mn --num_configs 512
```

Each output file is a YAML configuration containing both train and validation mixtures, along with metadata documenting the sampling hyperparameters and proxy model settings.

## Training Proxy Tokenizers

The tokenizer training pipeline lives in `train/train.py` and expects:

- A YAML configuration file that specifies the relative weighting of each domain under the
  `train` key.
- Text corpora stored in `train/<domain_name>` directories.

To launch a single experiment manually:

```bash
python3 -m train.train \
  --data_root /path/to/train \
  --output_dir /path/to/tokenizer \
  --cfg_file ./configs/domain_mix.yaml \
  --num_bytes 1073741824 \
  --vocab_size 65518 \
  --regex_string "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]?|\s*[\r\n]|\s+(?!\S)|\s+"
```

After training, run post-processing to build the Hugging Face tokenizer artifact:

```bash
python3 -m train.preprocess \
  --process_tgt /path/to/tokenizer \
  --output_dir ./tokenizers/example_tokenizer/post_processed
```

See [train_tokenizer.sh](https://github.com/jujeongho0/superTReXbpe/blob/main/train_tokenizer.sh). (`run_train config_mn {DATA_BYTES} {VOCAB_SIZE} trex_1gb_64k}`)

## Calculating Token Length Statistics

`calculate_length.py` computes the tokenization length distribution for every combination of trained tokenizer and dataset. Provide the tokenizer root directory, dataset root, and output path:

```bash
python3 -m train.calculate_length \
  --ds_root /path/to/valid \
  --tok_root /path/to/tokenizer \
  --out_pkl ./artifacts/token_lengths.pkl
```

The resulting pickle file contains a table where each row corresponds to a tokenizer and each column records the token lengths for a dataset. These statistics drive the regression code that estimates normalized-sequence-length efficiency across mixtures.

- **`train/trex_regression_model_train_infer.py`** – trains the regression model and predicts optimal mixture weights for new tokenizer candidates.

## Tokenizer training
Training a SuperBPE tokenizer involves two stages:

1. **Stage 1:** Learn subwords by enforcing whitespace pretokenization (equivalent to regular BPE training). See [train_tokenizer.sh](https://github.com/jujeongho0/superTReXbpe/blob/main/train_tokenizer.sh). (`run_train config_mn_optimal {DATA_BYTES} {VOCAB_SIZE} trex_30gb_128k}`)
2. **Stage 2:** Learn superwords by resuming tokenizer training, but this time skip the whitespace pretokenization step. See [extend_tokenizer.sh](https://github.com/jujeongho0/superTReXbpe/blob/main/extend_tokenizer.sh).

## Citation 
```
@misc{won2026trextokenizerregressionoptimal,
      title={TREX: Tokenizer Regression for Optimal Data Mixture}, 
      author={Inho Won and Hangyeol Yoo and Minkyung Cho and Jungyeul Park and Hoyun Song and KyungTae Lim},
      year={2026},
      eprint={2601.13588},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.13588}, 
}
```

```
@inproceedings{liu-etal-2025-superbpe,
  title={{SuperBPE}: Space travel for language models},
  author={Alisa Liu and Jonathan Hayase and Valentin Hofmann and Sewoong Oh and Noah A Smith and Yejin Choi},
  booktitle={Second Conference on Language Modeling},
  year={2025},
  url={https://arxiv.org/abs/2503.13423}
}
```

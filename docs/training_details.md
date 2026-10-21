# Training details

For the model, methodology, and published evaluations, see the [Qalb paper](https://arxiv.org/abs/2601.08141). This page covers only the files and practical limits of this repository.

## What is included

- `training/cpt_train.py`: the supplied full-precision CPT configuration, prepared as a paper-aligned candidate.
- `training/experiments/cpt_experiments.ipynb`: an alternative 4-bit CPT experiment.
- `presentation/qalb.html`: the standalone slide presentation.

The training code is based on files supplied with the project. It is not verified as the exact code used for the published run. SFT code, corpus preparation code, and the training corpus are not included.

## Training setup

The pinned environment is a reconstructed starting point for Linux x86_64, Python 3.11, and one NVIDIA GPU. It is not a recovered historical environment. Install it from the repository root with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The main script expects `urdu_data/llama31_8b_mixed_seq2048.bin` and a matching tokenizer in `local_tokenizer/`. The binary file must contain headerless `int32` token IDs in complete rows of 2,048 tokens. The corpus, tokenizer, and model weights are not included. The code checks token IDs and tokenizer compatibility before training.

Run the main configuration with:

```bash
python training/cpt_train.py
```

The notebook uses a separate default data path and prefers a local tokenizer, with a Hugging Face fallback. Review its `Config` before running. Keep different experiments in separate output directories so checkpoints and logs do not mix.

## Changes and limits

The repository version adds data and tokenizer checks, handles already-packed rows with the selected TRL/Unsloth APIs, supports fresh notebook runs and checkpoint resumption, and clears saved notebook outputs. The main script also keeps its configured embedding learning rate when the Trainer is initialized.

No GPU training was run during repository preparation. The pinned packages, model loading, CUDA behavior, checkpoint resume, and memory use still need a run on the target machine. Exact model and tokenizer revisions, the original corpus, and historical dependency versions are unavailable, so these files do not reproduce or validate the paper’s results. The paper’s SFT and benchmark evaluation stages are outside this repository.

The repository’s original code uses Apache 2.0. The presentation includes the Reveal.js runtime and its MIT license notice. Model, tokenizer, and Llama use remain subject to their upstream terms. Training data, weights, credentials, and run outputs are excluded from Git.

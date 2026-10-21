# Qalb-1.0

**An Urdu large language model built with Llama 3.1 8B.**

[Paper](https://arxiv.org/abs/2601.08141) · [Hugging Face model](https://huggingface.co/enstazao/Qalb-1.0-8B-Instruct) · [Training details](docs/training_details.md) · [Presentation](presentation/qalb.html) · [Apache 2.0 code license](LICENSE)

Qalb adapts an existing language model to Urdu through **continued pre-training (CPT)** followed by **supervised fine-tuning (SFT)**. This repository provides two author-supplied CPT implementations, their configurations, and practical documentation for research use. The released instruction model is available on Hugging Face.

**Paper:** *Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-Training*
Muhammad Taimoor Hassan, Jawad Ahmed, and Muhammad Awais · [arXiv:2601.08141](https://arxiv.org/abs/2601.08141)

> **Implementation status:** The main script closely matches the paper’s CPT configuration, but its identity as the exact historical training script has not been verified. The 4-bit notebook is an experimental alternative. Neither has been used here to reproduce the published results. SFT code, corpus preparation code, and the training corpus are not included.

## Model and training approach

The foundation is Llama 3.1 8B, a decoder-only Transformer for causal language modeling. The released Qalb checkpoint has undergone instruction tuning; the CPT script starts from the **base** checkpoint, `unsloth/Meta-Llama-3.1-8B`.

```text
Llama 3.1 8B base → Continued pre-training → Supervised fine-tuning → Qalb-1.0-8B-Instruct
                    Code provided here     Code not yet included
```

CPT extends language modeling on packed text. Each token sequence is supplied as both `input_ids` and `labels`; the causal language model shifts the targets internally to predict the next token. It does **not** use instruction–response pairs, a chat template, or assistant-only loss masking.

Both implementations use LoRA rather than updating every base-model parameter. The selected Unsloth release handles `embed_tokens` and `lm_head` as trainable saved modules alongside the attention/MLP adapters. Consequently, the trainable parameters include embedding/head weights; this is not full-parameter pre-training, nor an adapter-only configuration that leaves those weights frozen. Loading the main base model without 4-bit quantization does not imply full-parameter training.

Published benchmark results belong to the [paper’s evaluation](https://arxiv.org/html/2601.08141v1#S6), not to a new experiment performed for this repository.

## Repository layout

```text
Qalb-1.0/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── training/
│   ├── cpt_train.py
│   └── experiments/
│       └── cpt_experiments.ipynb
├── presentation/
│   └── qalb.html
└── docs/
    └── training_details.md
```

The standalone [presentation](presentation/qalb.html) is included with the repository. Open it in a browser; use the arrow keys or Space to navigate.

## Two CPT implementations

The values below are taken from the supplied code. They are not newly measured results.

| Configuration | Main script | Experimental notebook |
|---|---|---|
| File | `training/cpt_train.py` | `training/experiments/cpt_experiments.ipynb` |
| Provenance | Supplied as `cpt_test.py` | Supplied as `llama3.18b-cpt.ipynb` |
| Status | Paper-aligned candidate | Alternative quantized CPT |
| Base loading | Unquantized; automatic dtype, intended BF16 on A100 | 4-bit base; intended BF16 compute |
| LoRA rank / alpha / dropout | 128 / 32 / 0 | 128 / 32 / 0 |
| Learning rate / embedding LR | 2e-5 / 2e-6 | 2e-5 / 2e-6 |
| Sequence length | 2,048 | 2,048 |
| Per-device batch × accumulation | 16 × 8 | 1 × 16 |
| Effective batch, one GPU | 128 sequences | 16 sequences |
| Epochs | 1 | 1 |
| Evaluation split | 0.2%, maximum 4,096 rows | 0.5%, no explicit cap |
| Evaluation / checkpoint interval | 2,000 optimizer steps | 1,000 optimizer steps |
| Tokenizer | Required local snapshot | Local preferred; Hub fallback |
| Logging | TensorBoard, CSV, generated samples | CSV; final generation sample |

Both use AdamW 8-bit, cosine scheduling, warmup ratio 0.05, seed 3407, and Unsloth gradient checkpointing. Both target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens`, and `lm_head`.

## Installation

The candidate environment targets **Linux x86_64, Python 3.11, and one NVIDIA A100 80 GB**. macOS/CPU execution is not a supported training configuration. The selected Unsloth version excludes Python 3.13 and later.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
```

The pinned Torch 2.6.0 Linux wheels use CUDA 12.4 dependencies; the GPU driver must support that runtime. The pins are a **reconstructed candidate environment**, not a recovered historical lockfile. Key API sources and dependency metadata were reviewed, but CUDA installation and training have not been tested here. Transitive dependencies are not fully locked. See [training details](docs/training_details.md) before changing versions.

Authenticate using `HF_TOKEN` supplied through your environment or a secret manager. For an interactive terminal without putting the token in shell history:

```bash
read -r -s -p 'HF_TOKEN: ' HF_TOKEN
export HF_TOKEN
printf '\n'
```

Accept any model access terms required by the checkpoint provider. The code does not load `.env` files automatically.

## Dataset preparation

Prepare the token file separately, with the tokenizer snapshot used to encode it. This repository does not supply the original scraping, cleaning, packing pipeline, corpus, or a substitute dataset.

The main script expects:

```text
urdu_data/llama31_8b_mixed_seq2048.bin
local_tokenizer/                          # saved Llama 3.1 tokenizer files
```

Requirements:

- A flat, headerless array of native `int32` token IDs; little-endian on the target Linux x86_64 host.
- Complete rows of **2,048 tokens**, with at least two rows for a nonempty train/evaluation split.
- File size divisible by `2048 × 4 = 8192` bytes.
- Token IDs from the same tokenizer vocabulary as the base model. Preserve the original packing and BOS/EOS policy; the code reports a BOS-at-row-start check but does not enforce one.
- No padded positions requiring loss masking: both implementations use all-one attention masks and supervise every token position except the model’s internal causal shift.

The added checks scan all token IDs, compare local and model tokenizer mappings, and check embedding bounds. They do not prove corpus provenance or recover the tokenizer that originally produced a `.bin` file. Keep corpus/tokenizer checksums and model revision IDs with each experiment.

## Running CPT

Review the `Config` dataclass in the chosen implementation. Paths are relative to the working directory; run the script from the repository root.

```bash
python training/cpt_train.py
```

There is no command-line argument parser. Set paths and experimental settings in `Config` deliberately, and retain the configuration with your run record. A memory reduction such as lowering the batch size is a new configuration, not an unchanged reproduction.

For the experimental notebook:

```bash
jupyter lab training/experiments/cpt_experiments.ipynb
```

Its first code cell finds the repository root and installs `requirements.txt`. Restart the kernel after installation, then run the cells in order. Its preserved default corpus path is `llama31_8b_mixed_tokens/llama31_8b_mixed_seq2048.bin`. It retains a local-first tokenizer with Hub fallback; the fallback does not establish which tokenizer created your data.

## Checkpointing and resuming

Both implementations look for the latest `checkpoint-*` directory inside `Config.output_dir`. They resume when one exists and start fresh when none exists. They retain the latest two checkpoints. The script checkpoints every 2,000 optimizer steps; the notebook every 1,000. A shorter run may finish before producing an intermediate checkpoint.

Rerun the script or notebook training cell to resume. Do not change the data, tokenizer, model, split, optimizer settings, or batch settings when continuing a run. `num_train_epochs=1` is the total target, not one additional epoch on every restart. A final adapter export in `urdu_cpt_lora/` is not a substitute for a full Trainer checkpoint containing optimizer, scheduler, and trainer state.

The two implementations share default output names. Use separate `output_dir`, CSV paths, and adapter-export directories for different experiments. Existing CSV and generation files are appended to; selecting a new output directory alone does not isolate those other paths.

## Monitoring training

```bash
tensorboard --logdir outputs_urdu_cpt/tb_logs
```

The main script writes:

| Artifact | Contents |
|---|---|
| `outputs_urdu_cpt/run_info.txt` | Configuration, row counts, parameter counts, and environment information |
| `outputs_urdu_cpt/tb_logs/` | TensorBoard events |
| `training_logs.csv` | Step, training loss, evaluation loss, and `exp(eval_loss)` |
| `outputs_urdu_cpt/generations/1_generation.txt` | A sampled evaluation prompt and completion after each evaluation |
| `outputs_urdu_cpt/checkpoint-*/` | Resumable Trainer checkpoints |
| `urdu_cpt_lora/` | Final adapter export and tokenizer |

Perplexity is derived from token-level evaluation loss; it is not an instruction benchmark score. Generation examples are qualitative checks. The CSV may contain repeated evaluation rows because final evaluation is logged through both the callback and explicit final reporting. Adapter exports can be substantial when embedding/head modules are saved.

See the [review record](docs/training_details.md#changes-from-the-supplied-files) for changes, validation scope, callback behavior, and remaining reproducibility limits.

## Citation

```bibtex
@article{hassan2026qalb,
  title   = {Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-Training},
  author  = {Hassan, Muhammad Taimoor and Ahmed, Jawad and Awais, Muhammad},
  journal = {arXiv preprint arXiv:2601.08141},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.08141},
  url     = {https://arxiv.org/abs/2601.08141}
}
```

[CITATION.cff](CITATION.cff) supplies machine-readable citation metadata. The citation above uses the supplied arXiv record.

## License

The repository’s original code is released under [Apache License 2.0](LICENSE), as selected by the project maintainer. This license does not relicense model weights, tokenizers, third-party software, the paper, or source datasets.

**Built with Llama.** Consult the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) and applicable model terms for use or redistribution of Llama-derived artifacts. The [Qalb model card](https://huggingface.co/enstazao/Qalb-1.0-8B-Instruct) currently lists Apache 2.0 metadata; that label should not be treated as replacing upstream obligations. No model weights or training data are distributed in this repository.

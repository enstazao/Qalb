# Training details and review record

This document separates the paper’s methodology, the supplied implementations, and changes made while preparing this repository. Review date: 21 October 2026.

## Scope and provenance

`training/cpt_train.py` is derived from the author-supplied `cpt_test.py`. It is a **paper-aligned CPT candidate**, not a verified archival copy of the exact published-run script. `training/experiments/cpt_experiments.ipynb` is derived from `llama3.18b-cpt.ipynb` and is an **experimental quantized CPT configuration**. Neither file establishes that a new run reproduces the paper.

The package versions in `requirements.txt` form a **reconstructed candidate environment**. No historical environment lockfile, exact model/tokenizer revisions, original corpus, or run checkpoint was supplied. SFT is part of the Qalb model’s described development process, but no SFT implementation is included here.

Source-file SHA-256 hashes at repository preparation:

```text
cpt_test.py
5e589712d10b4ed6435c43ab59d93dc9bf9de3eec4a98e09d18bf5997d04ba28

llama3.18b-cpt.ipynb
1204ee70a992b2c7928691cd3db8b26ac15e9bb9b690a622cca5b40e7c815a4b
```

These hashes identify the locally supplied inputs, not the published training artifacts. The source files outside this repository were preserved.

## Objective and trainable parameters

Both implementations perform causal next-token prediction on fixed-length token rows. The datasets return identical initial values for `input_ids` and `labels`, with independent tensors, and all-one attention masks. The model performs the causal label shift. There is no instruction template, prompt/response partition, or assistant-only loss mask.

The main script sets `load_in_4bit=False`. The notebook uses `unsloth/Meta-Llama-3.1-8B-bnb-4bit` with `load_in_4bit=True`. Both leave the loading dtype as `None`, allowing Unsloth to choose it, and request BF16 training on supported hardware. Neither implementation updates all base weights.

In the selected Unsloth source, `embed_tokens` and `lm_head` targets are handled as `modules_to_save`, with trainable embedding/head weights alongside low-rank attention/MLP adapters. The specialized optimizer routes trainable names ending in `modules_to_save.default.weight` to the embedding learning rate. Do not describe this configuration as pure LoRA matrices alone, or assume the saved adapter directory will be small. Parameter counts are measured by the main script at runtime; no parameter count was independently measured during this review.

## Configuration comparison

All values below were read from the supplied implementations and retained.

| Setting | Paper-aligned script | Quantized notebook |
|---|---|---|
| Base model | `unsloth/Meta-Llama-3.1-8B` | `unsloth/Meta-Llama-3.1-8B-bnb-4bit` |
| 4-bit loading | False | True |
| Sequence length | 2048 | 2048 |
| LoRA rank / alpha / dropout | 128 / 32 / 0.0 | 128 / 32 / 0.0 |
| LoRA bias | `none` | `none` |
| Learning rate | 2e-5 | 2e-5 |
| Embedding learning rate | 2e-6 | 2e-6 |
| Weight decay | 0.0 | 0.0 |
| Optimizer | `adamw_8bit` | `adamw_8bit` |
| Scheduler / warmup ratio | cosine / 0.05 | cosine / 0.05 |
| Train batch / accumulation | 16 / 8 | 1 / 16 |
| Effective batch on one GPU | 128 sequences | 16 sequences |
| Evaluation batch | 4 | 1 |
| Epochs | 1.0 | 1.0 |
| BF16 / FP16 configuration | True / False | True / False |
| Evaluation fraction | 0.002 | 0.005 |
| Evaluation cap | 4096 rows | None |
| Evaluation / save interval | 2000 / 2000 steps | 1000 / 1000 steps |
| Logging interval | 50 steps | 50 steps |
| Checkpoint retention | 2 | 2 |
| Reporting | TensorBoard | `none` (CSV callback still active) |
| Seed | 3407 | 3407 |

Targets in both implementations: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `embed_tokens`, `lm_head`.

The paper’s [Table II](https://arxiv.org/html/2601.08141v1#S4.T2) supports the main configuration’s rank, alpha, targets, optimizer, rates, effective batch, sequence length, and precision. Evaluation split, callback settings, checkpoint interval, and other code-specific choices are reported here as properties of the supplied files, not inferred paper facts.

## Changes from the supplied files

No original `Config` field value, LoRA target, learning rate, batch size, accumulation count, seed, epoch count, or split fraction was changed. The following changes affect validation, compatibility, or error handling and should be retained in any reproducibility record.

| Change | Implementation | Reason and behavior |
|---|---|---|
| Rename and organize | Both | `cpt_test.py` → `training/cpt_train.py`; notebook → `training/experiments/cpt_experiments.ipynb`. |
| Import Unsloth before Transformers | Both | Apply Unsloth patches before the affected libraries are imported. |
| Explicit packed-dataset handling | Both | Set `max_seq_length=cfg.seq_len`, `packing=False`, and `dataset_kwargs={"skip_prepare_dataset": True}`. TRL must not preprocess or repack existing 2048-token rows. The dataset also implements the batched, side-effect `.map()` call that Unsloth may use to count token IDs during its startup check. |
| Preserve the embedding LR after Trainer construction | Both | The reviewed Unsloth compatibility wrapper can rebuild arguments as `SFTConfig`, dropping extension-only fields. Reassign `trainer.args.embedding_learning_rate` from the unchanged configuration before optimizer creation. This prevents silent fallback to a single learning rate. |
| Validate every packed token ID | Both | Reject incomplete/empty files, fewer than two rows, negative IDs, and IDs absent from the local tokenizer. Scan in bounded chunks without editing or filtering data. Adds a read pass before training. |
| Validate model/tokenizer mappings | Both | Compare full token-to-ID dictionaries and BOS/EOS IDs; check vocabulary and special IDs against input/output embedding rows. Replace the script’s disabled size-only check. Fail on differences; never resize embeddings automatically. |
| Fresh-run notebook support | Notebook | Replace unconditional `resume_from_checkpoint=True` with latest-checkpoint detection. Fresh runs now start without an existing checkpoint; resumed runs still use one. |
| Propagate checkpoint-discovery errors | Script | Remove the catch-and-start-fresh fallback. A discovery error now stops execution rather than silently starting a different run. |
| Restore generation mode on failure | Script | Wrap callback generation/decoding in `try/finally` and restore the prior ordinary train/eval flag, including when generation raises. Sampling settings are unchanged. |
| Update evaluation keyword | Notebook | Use `eval_strategy`, the keyword supported in the selected Transformers version, instead of its deprecated alias. Correct the contradictory comment in the script. |
| Repository-aware environment setup | Notebook | Replace open-ended minimum-version installation with the shared pinned requirements and establish the repository root as the working directory. |
| Sanitize notebook | Notebook | Clear all saved outputs, execution counts, and cell metadata. Remove the unverified Python 3.14 kernel-version metadata. Add a concise experimental-status introduction. |
| Correct comments | Script | Remove the claim that embedding LR is mostly a no-op and accurately describe the active vocabulary checks. |

These are maintenance changes to the supplied candidate implementations. They are not evidence that the modified code is bit-for-bit equivalent to an unknown historical run. In particular, restoring an embedding LR that a different library stack dropped can change optimization relative to that broken stack.

## Dependency and API review

The candidate stack pins Python-package releases around Unsloth 2025.3.19: Torch 2.6.0, torchvision 0.21.0, xFormers 0.0.29.post3, Triton 3.2.0, Unsloth Zoo 2025.3.17, Transformers 4.49.0, TRL 0.15.2, PEFT 0.14.0, Accelerate 1.4.0, and bitsandbytes 0.45.3. The complete direct pin list is in [requirements.txt](../requirements.txt).

This is a versioned starting point for testing, not a claim that these versions were used for the paper or that they are the latest releases. All 24 specified releases and their pairwise declared constraints were checked using PyPI metadata for Linux/Python 3.11. This does not replace a complete dependency-resolution, installation, vulnerability, or CUDA execution test. Transitive dependencies are not fully locked.

Source review established:

- The selected Unsloth release requires Python `<3.13,>=3.9`, supports Transformers `>=4.46.1` excluding 4.47.0, and limits TRL to `<=0.15.2` with additional exclusions. Python 3.11 is the documented target.
- xFormers 0.0.29.post3 requires Torch 2.6.0; torchvision 0.21.0 requires the same Torch version. Torch 2.6.0 Linux x86_64 metadata uses CUDA 12.4 dependencies and Triton 3.2.0.
- `UnslothTrainingArguments` extends TRL’s `SFTConfig`; `UnslothTrainer` extends `SFTTrainer`. The class name does not change the objective: the input data and loss remain plain causal language modeling here.
- `eval_strategy` is supported by Transformers 4.49.0. The selected TRL/Unsloth compatibility layer accepts the existing `tokenizer=` argument and forwards it to `processing_class` where needed.
- Skipping TRL dataset preparation is necessary for these custom PyTorch datasets. The reviewed TRL preprocessing otherwise accesses Hugging Face dataset metadata and mapping APIs. Unsloth may separately call `train_dataset.map(..., batched=True)` to count token IDs when its untrained-token check finds repeated embedding rows; both packed datasets implement that narrow side-effect scan. The actual Unsloth training hook has not been executed on the target stack.
- The selected Unsloth optimizer’s embedding-group selection is based on trainable saved-module parameter names. The main script’s old “mostly no-op” comment was misleading for this target list.

Upstream references: [Unsloth 2025.3.19 release](https://pypi.org/project/unsloth/2025.3.19/), [Unsloth Zoo 2025.3.17](https://pypi.org/project/unsloth-zoo/2025.3.17/), [Transformers 4.49.0 arguments](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/training_args.py), [TRL 0.15.2 trainer](https://github.com/huggingface/trl/blob/v0.15.2/trl/trainer/sft_trainer.py), and the versioned Python wheels published on PyPI. Unsloth’s `trainer.py` and `models/llama.py` were inspected from its published wheel.

Before a full run on the target GPU, run `python -m pip check`, verify CUDA/BF16 availability, and exercise a small disposable run covering model loading, optimizer creation, evaluation, checkpoint saving/resumption, and final export. Such a smoke run is a separate diagnostic configuration, not a reported Qalb result. Record the final installed environment:

```bash
python -m pip freeze > environment.lock.txt
```

Also record driver/CUDA versions, model revision, tokenizer files, token-file checksum, code commit, and output paths. The generated environment file is ignored until intentionally reviewed for release.

## Tokenizer and dataset review

The actual tokenizer directory and packed corpus were not supplied, so their real compatibility has **not** been established. The added checks verify mappings at runtime against the tokenizer returned with the loaded base model. They do not identify which tokenizer historically created a token file. Equal vocabulary sizes alone are insufficient: `vocab_size` may exclude added tokens, and equal-sized vocabularies can map text to different IDs.

The script uses only `./local_tokenizer`; it has no Hub fallback. The notebook prefers the local directory and otherwise loads a Hub tokenizer. Its original optional local/Hub sample-comparison branch normally cannot run because the loader only sets one of those objects. It is preserved as an original-code limitation; the new direct model-tokenizer comparison provides the active check.

`np.int32` uses native byte order. The expected deployment is little-endian Linux x86_64. Each row has 2048 IDs and no header. The training label tensor copies the unshifted IDs, and the collator stacks rows without padding. A different packing policy, inserted padding, or tokenizer revision changes the training inputs even if every ID passes validation.

Split construction shuffles row indices with seed 3407, takes evaluation rows first, and uses the remainder for training. The script caps evaluation at 4096 rows. This is a row-level split: neighboring rows from the same source document can land in both partitions. Document-level separation is not enforced or reconstructed here.

## Checkpoints and callbacks

**Resumption.** Both implementations select the numerically latest checkpoint folder through `get_last_checkpoint`. This selects a folder, not a verified complete checkpoint. A damaged latest checkpoint will fail rather than automatically rolling back to an older one. The script now propagates discovery failures. The notebook can start fresh when no checkpoint exists. Actual checkpoint serialization/reloading with Unsloth has not been exercised here.

**Output isolation.** Defaults such as `outputs_urdu_cpt`, `training_logs.csv`, and `urdu_cpt_lora` are shared. Do not mix the unquantized and quantized runs under the same names. Change all relevant output/export paths for a new experiment, and never resume a checkpoint against changed data or training settings.

**CSV.** Logging appends a header only when creating a new file. Training loss and evaluation loss can occur on separate rows; absent values are empty fields. Evaluation perplexity is `exp(eval_loss)`. The original explicit final report can duplicate the evaluation entry already written by the callback. Extremely large losses can overflow `math.exp`; nonfinite-loss handling was not redesigned.

**Generation.** After each main-script evaluation, the callback samples one evaluation row, uses its first 128 tokens as the prompt, and generates up to 120 new tokens with temperature 0.8 and top-p 0.9. It appends prompt/completion text to a local file. This is qualitative inspection, not a benchmark evaluation. It consumes the Python/Torch random streams; RNG isolation was intentionally not added because that would change the original sampling/training trajectory. Changing evaluation cadence can therefore affect the sequence of random draws.

The callback restores the ordinary train/eval mode on both success and failure. Unsloth’s internal inference/training switches and gradient-checkpointing state still require validation on the target GPU stack. Generation errors propagate; samples are not silently skipped. These callbacks were written for one process, have no distributed write guards, and are not presented as multi-GPU-safe.

The notebook’s final generation sample follows its evaluation cell; it has no equivalent per-evaluation generation callback. Both final example prompts request news-like text without a retrieval source, so completions are not evidence of factual accuracy.

## Validation performed

The local host is macOS without a CUDA device. Training dependencies were inspected as source/metadata rather than imported as an operational CUDA environment.

- Parsed the main script and notebook Python cells; validated the notebook with `nbformat`.
- Compared every original and repository `Config` field: no values changed.
- Ran CPU component checks for packed rows, independent label tensors, and the fixed-length collator.
- Checked deterministic, disjoint row splits and rejection of malformed/empty/one-row data, negative IDs, and unknown IDs.
- Checked tokenizer ID-map mismatch, BOS/EOS mismatch, and embedding-bound rejection using small test fixtures.
- Exercised CSV append/reopen behavior and perplexity calculation.
- Exercised generation logging and mode restoration with a mock model, including a generation failure.
- Exercised notebook fresh-start and checkpoint-resume dispatch with a stub Trainer.
- Confirmed notebook and script use the same validation functions.
- Reviewed the 24 direct dependency pins against published package metadata and checked declared constraints among those pins.

CPU checks used the host’s Python 3.13.9, Torch 2.12.1, and NumPy 2.3.5 on extracted pure definitions, not the proposed training environment. The actual Unsloth Trainer, optimizer, model weights, corpus, CUDA kernels, checkpoints, and GPU memory footprint were not tested. These checks establish limited component behavior, not successful training or benchmark reproduction.

## Security and release scope

Both original inputs and the prepared repository were inspected for common Hugging Face, OpenAI, GitHub, AWS, and Google credential formats, private-key headers, credential assignments, local absolute paths, and notebook outputs. No embedded credentials or private absolute filesystem paths were found in the supplied code. The notebook contained one saved stream output; all outputs and execution counts were removed from the repository copy.

Authentication uses `HF_TOKEN` from the environment; the notebook also retains its hidden-input prompt. Hugging Face `login` may persist credentials in the user’s local Hugging Face cache, even with Git credential storage disabled. Do not copy that cache into the repository. Neither script requires OpenAI, GitHub, AWS, or Google credentials.

`.gitignore` excludes credentials, local tokenizer snapshots, packed data, model weights, checkpoints, generated text, TensorBoard events, CSV logs, Python caches, and notebook checkpoint directories. No weights, datasets, generated samples, slides, or personal source files outside the requested repository are included. This inspection is a scoped secret/content scan, not a guarantee that all possible secrets or dependency vulnerabilities have been detected.

## Remaining reproducibility limits

1. Exact historical script identity and dependency versions are unverified.
2. Original corpus, tokenizer snapshot, packing implementation, and artifact checksums are unavailable in this repository.
3. Model IDs are not pinned to recovered historical Hub commit hashes; record the resolved revisions for any new run.
4. GPU memory fit, BF16 dtype selection, actual optimizer groups, and full train/evaluate/resume behavior remain to be tested on the selected environment.
5. SFT and the paper’s benchmark evaluation pipeline are not included, so CPT alone cannot reproduce the released instruct model or its reported scores.
6. Dataset splitting, callback RNG consumption, duplicate final CSV rows, and single-process assumptions are retained and documented above.

## Sources

- [Qalb paper, arXiv](https://arxiv.org/abs/2601.08141)
- [Paper methodology and training tables](https://arxiv.org/html/2601.08141v1#S4)
- [Released Qalb instruction model](https://huggingface.co/enstazao/Qalb-1.0-8B-Instruct)
- [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)
- Versioned upstream API/package sources linked in the dependency review above.

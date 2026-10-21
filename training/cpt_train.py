#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Llama-3.1-8B Urdu Continued Pretraining (CPT) with Unsloth + LoRA

- Assumes:
    - Packed token file at: urdu_data/llama31_8b_mixed_seq2048.bin
      (int32, rows of length 2048)
    - Local tokenizer directory: ./local_tokenizer
    - GPU: A100-80GB (bf16 supported)
- Performs:
    - Loads tokenizer from local directory ONLY (no HF fallback)
    - Builds Dataset from packed .bin
    - Loads full bf16 base model (unsloth/Meta-Llama-3.1-8B), no 4-bit
    - Checks token mappings, embedding bounds, and packed token IDs
    - Attaches LoRA (r=128, alpha=32, dropout=0.0) on selected modules
    - Trains for exactly 1 epoch over train rows
    - Uses AdamW-8bit, bf16, gradient checkpointing
    - Logs to TensorBoard and CSV
    - Saves LoRA adapters + tokenizer
    - Runs final evaluation + small generation check
    - AFTER EACH EVAL: generates one sample from a random eval row and
      appends it to outputs_urdu_cpt/generations/1_generation.txt
"""

import os
import math
import gc
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Import Unsloth before Transformers so its patches are applied first.
import unsloth

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import login

from unsloth import (
    FastLanguageModel,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bfloat16_supported,
)

# -------------------------------------------------------------------------
# 1. Basic setup: TF32 + seeds
# -------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


@dataclass
class Config:
    """Configuration for Urdu CPT run."""
    # Data (packed token file)
    bin_path: str = "urdu_data/llama31_8b_mixed_seq2048.bin"
    seq_len: int = 2048
    eval_ratio: float = 0.002          # 0.2% rows for eval (capped later)
    shuffle_indices: bool = True

    # Tokenizer
    local_tokenizer_dir: str = "./local_tokenizer"

    # Base model (full, not 4-bit)
    base_model_id: str = "unsloth/Meta-Llama-3.1-8B"

    # LoRA (CPT)
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    )
    bias: str = "none"

    # Optimization
    learning_rate: float = 2e-5
    embedding_learning_rate: float = 2e-6  # LR for trainable embedding/head weights
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # Batching / precision for A100-80GB
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 8  # effective batch = 128 sequences
    per_device_eval_batch_size: int = 4
    bf16: bool = True
    fp16: bool = False

    # Training length: strict 1 epoch
    num_train_epochs: float = 1.0

    # Checkpointing / logging / storage
    output_dir: str = "outputs_urdu_cpt"
    logging_steps: int = 50
    eval_steps: int = 2000
    save_steps: int = 2000
    save_total_limit: int = 2
    report_to: Optional[str] = "tensorboard"
    csv_log_path: str = "training_logs.csv"


class TokenBinDataset(Dataset):
    """
    Memory-mapped dataset over packed token file.
    - bin_path: path to .bin (int32)
    - seq_len:  sequence length (2048)
    - rows:     list of row indices to expose
    """

    def __init__(self, bin_path: str, seq_len: int, rows: Optional[List[int]] = None):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.mm = np.memmap(bin_path, dtype=np.int32, mode="r")
        if self.mm.size % seq_len != 0:
            raise ValueError(f"Bin size {self.mm.size} not divisible by seq_len {seq_len}.")
        self.n_rows = self.mm.size // seq_len
        self.rows = rows if rows is not None else list(range(self.n_rows))

    def __len__(self):
        return len(self.rows)

    def map(self, function, batched=False, batch_size=1000, desc=None, **kwargs):
        """Apply a batched side-effect transform used by Unsloth's token checks."""
        if not batched:
            raise NotImplementedError("TokenBinDataset.map supports batched transforms only.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            rows = []
            for index in range(start, stop):
                row = self.rows[index]
                offset = row * self.seq_len
                rows.append(self.mm[offset:offset + self.seq_len].tolist())
            function({"input_ids": rows})
        return self

    def __getitem__(self, idx):
        r = self.rows[idx]
        start = r * self.seq_len
        end = start + self.seq_len
        ids = np.array(self.mm[start:end], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(ids),
            "labels": torch.from_numpy(ids.copy()),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


class CSVLoggerCallback(TrainerCallback):
    """Simple CSV logger for loss, eval_loss, and perplexity."""

    def __init__(self, path: str):
        self.path = path
        self._f = None

    def _open(self):
        if self._f is None:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            file_exists = os.path.exists(self.path)
            self._f = open(self.path, "a", encoding="utf-8")
            if not file_exists:
                self._f.write("step,loss,eval_loss,eval_perplexity\n")
                self._f.flush()

    def on_train_begin(self, args, state, control, **kwargs):
        self._open()

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._open()
        logs = logs or {}
        step = state.global_step
        loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        ppl = math.exp(eval_loss) if eval_loss is not None else None
        line = (
            f"{step},"
            f"{'' if loss is None else loss},"
            f"{'' if eval_loss is None else eval_loss},"
            f"{'' if ppl is None else ppl}\n"
        )
        self._f.write(line)
        self._f.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._f:
            self._f.close()
            self._f = None


# -------------------------------------------------------------------------
# Generation callback: one sample after each eval
# -------------------------------------------------------------------------
class GenerationCallback(TrainerCallback):
    """
    After each evaluation, generate from ONE random eval row and
    append to outputs_urdu_cpt/generations/1_generation.txt
    """

    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        generations_path: str,
        max_prompt_tokens: int = 128,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.generations_path = generations_path
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        gen_dir = os.path.dirname(self.generations_path)
        if gen_dir:
            os.makedirs(gen_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return
        if len(self.eval_dataset) == 0:
            return

        model_device = next(model.parameters()).device

        # Pick ONE random row from eval set
        idx = random.randint(0, len(self.eval_dataset) - 1)
        row = self.eval_dataset[idx]

        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]

        # Truncate prompt tokens
        prompt_ids = input_ids[: self.max_prompt_tokens].unsqueeze(0).to(model_device)
        prompt_mask = attention_mask[: self.max_prompt_tokens].unsqueeze(0).to(model_device)

        prompt_len = prompt_ids.shape[1]

        model_was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Separate prompt and completion
            new_tokens = gen_ids[0, prompt_len:]
            prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            completion_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        finally:
            model.train(model_was_training)

        # Append to single file 1_generation.txt
        with open(self.generations_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"STEP {state.global_step} | eval row idx {idx}\n")
            f.write("-" * 80 + "\n")
            f.write("PROMPT:\n")
            f.write(prompt_text.strip() + "\n\n")
            f.write("COMPLETION:\n")
            f.write(completion_text.strip() + "\n\n")

        print(
            f"[GenerationCallback] Wrote sample for step {state.global_step} "
            f"from eval row {idx} to {self.generations_path}"
        )


# -------------------------------------------------------------------------
# Tokenizer utilities: strict *local* load (no HF)
# -------------------------------------------------------------------------
def load_tokenizer(cfg: Config) -> AutoTokenizer:
    """Load tokenizer from local directory ONLY. No HF fallback."""
    if not os.path.isdir(cfg.local_tokenizer_dir):
        raise RuntimeError(
            f"Expected local tokenizer dir at {cfg.local_tokenizer_dir}, "
            "but it does not exist. Put the tokenizer there."
        )

    print(f"Found local tokenizer dir: {cfg.local_tokenizer_dir}")
    tok = AutoTokenizer.from_pretrained(
        cfg.local_tokenizer_dir,
        use_fast=True,
        local_files_only=True,
    )
    print("Loaded local tokenizer.")

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        print("pad_token was None; set pad_token = eos_token.")

    return tok


# -------------------------------------------------------------------------
# Dataset builder
# -------------------------------------------------------------------------
def validate_packed_tokens(cfg, tokenizer):
    """Check file layout and token IDs without modifying the corpus."""
    size = os.path.getsize(cfg.bin_path)
    row_bytes = np.dtype(np.int32).itemsize * cfg.seq_len
    if size % row_bytes or size < 2 * row_bytes:
        raise ValueError("Packed data must contain at least two complete int32 token rows.")
    valid_ids = set(tokenizer.get_vocab().values())
    if not valid_ids or min(valid_ids) < 0:
        raise ValueError("Tokenizer vocabulary must contain non-negative token IDs.")
    # Check all IDs in bounded chunks; no token replacement or filtering.
    max_id = max(valid_ids)
    allowed = np.zeros(max_id + 1, dtype=np.bool_)
    allowed[list(valid_ids)] = True
    packed = np.memmap(cfg.bin_path, dtype=np.int32, mode="r")
    for start in range(0, packed.size, 1_048_576):
        chunk = packed[start:start + 1_048_576]
        if chunk.min() < 0 or chunk.max() > max_id or not allowed[chunk].all():
            raise ValueError("Packed data contains token IDs outside the tokenizer vocabulary.")
    del packed


def validate_tokenizer_model(tokenizer, model_tokenizer, model):
    """Reject incompatible token mappings before adapter attachment."""
    vocab = tokenizer.get_vocab()
    if vocab != model_tokenizer.get_vocab():
        raise ValueError("Local and model token-to-ID mappings differ. Use the tokenizer used for packing.")
    for name in ("bos_token_id", "eos_token_id"):
        if getattr(tokenizer, name) != getattr(model_tokenizer, name):
            raise ValueError(f"Tokenizer {name} differs from the model tokenizer.")
    input_rows = model.get_input_embeddings().weight.shape[0]
    output_rows = model.get_output_embeddings().weight.shape[0]
    ids = list(vocab.values()) + list(tokenizer.all_special_ids)
    if not ids or min(ids) < 0 or max(ids) >= min(input_rows, output_rows):
        raise ValueError("Tokenizer IDs exceed the model embedding or output vocabulary.")
    print("Tokenizer mappings and embedding bounds checked; no vocabulary resizing performed.")


def build_train_eval_from_bin(cfg: Config, tokenizer: AutoTokenizer):
    """Build train and eval datasets from packed .bin file."""
    if not os.path.isfile(cfg.bin_path):
        raise FileNotFoundError(f"Packed .bin not found at {cfg.bin_path}")

    validate_packed_tokens(cfg, tokenizer)

    mm = np.memmap(cfg.bin_path, dtype=np.int32, mode="r")
    n_rows_total = mm.size // cfg.seq_len
    del mm
    gc.collect()

    idx = np.arange(n_rows_total)
    if cfg.shuffle_indices:
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)

    n_eval = max(1, int(n_rows_total * cfg.eval_ratio))
    n_eval = min(n_eval, 4096)  # cap eval rows to 4k
    eval_rows = idx[:n_eval].tolist()
    train_rows = idx[n_eval:].tolist()

    train_ds = TokenBinDataset(cfg.bin_path, cfg.seq_len, train_rows)
    eval_ds = TokenBinDataset(cfg.bin_path, cfg.seq_len, eval_rows)

    tokens_per_epoch = len(train_rows) * cfg.seq_len
    print(
        f"Bin loaded. Total rows: {n_rows_total:,} | "
        f"Train rows: {len(train_rows):,} | Eval rows: {len(eval_rows):,}"
    )
    print(f"Tokens per epoch (train only): {tokens_per_epoch:,}")

    bos_id = tokenizer.bos_token_id
    sample = min(200, len(train_ds))
    if bos_id is not None and sample > 0:
        hits = 0
        for i in range(sample):
            if train_ds[i]["input_ids"][0].item() == bos_id:
                hits += 1
        print(f"BOS@0: {hits}/{sample} train rows start with BOS.")

    return train_ds, eval_ds


def collate_fixed_rows(batch):
    """Collator for fixed-length rows (no padding)."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# -------------------------------------------------------------------------
# Utility: parameter counting + run info logging
# -------------------------------------------------------------------------
def count_parameters(model: torch.nn.Module):
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


def write_run_info(
    cfg: Config,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
    model: torch.nn.Module,
    training_args: UnslothTrainingArguments,
):
    """Write key run info to a text file (not huge)."""
    run_info_path = os.path.join(cfg.output_dir, "run_info.txt")
    total_params, trainable_params = count_parameters(model)

    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    n_train_rows = len(train_ds)
    n_eval_rows = len(eval_ds)
    tokens_per_epoch = n_train_rows * cfg.seq_len
    steps_per_epoch = math.ceil(n_train_rows / effective_batch)

    with open(run_info_path, "w", encoding="utf-8") as f:
        f.write("=== Urdu CPT Run Info ===\n\n")
        f.write(f"Base model: {cfg.base_model_id}\n")
        f.write(f"Bin path: {cfg.bin_path}\n")
        f.write(f"Local tokenizer dir: {cfg.local_tokenizer_dir}\n\n")

        f.write("=== Tokenizer ===\n")
        f.write(f"Vocab size: {tokenizer.vocab_size}\n")
        f.write(f"BOS token id: {tokenizer.bos_token_id}\n")
        f.write(f"EOS token id: {tokenizer.eos_token_id}\n\n")

        f.write("=== Data ===\n")
        f.write(f"Sequence length: {cfg.seq_len}\n")
        f.write(f"Train rows: {n_train_rows}\n")
        f.write(f"Eval rows: {n_eval_rows}\n")
        f.write(f"Tokens per epoch (train): {tokens_per_epoch}\n\n")

        f.write("=== Training setup ===\n")
        f.write(f"Num train epochs: {cfg.num_train_epochs}\n")
        f.write(f"Per-device train batch size: {cfg.per_device_train_batch_size}\n")
        f.write(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}\n")
        f.write(f"Effective batch (sequences): {effective_batch}\n")
        f.write(f"Approx steps per epoch: {steps_per_epoch}\n")
        f.write(f"Learning rate (LoRA): {cfg.learning_rate}\n")
        f.write(f"Embedding LR: {cfg.embedding_learning_rate}\n")
        f.write(f"Warmup ratio: {cfg.warmup_ratio}\n")
        f.write(f"LR scheduler: {cfg.lr_scheduler_type}\n")
        f.write(f"Weight decay: {cfg.weight_decay}\n")
        f.write(f"bf16: {cfg.bf16}\n")
        f.write(f"fp16: {cfg.fp16}\n")
        f.write(f"Optim: {training_args.optim}\n\n")

        f.write("=== LoRA ===\n")
        f.write(f"LoRA r: {cfg.lora_r}\n")
        f.write(f"LoRA alpha: {cfg.lora_alpha}\n")
        f.write(f"LoRA dropout: {cfg.lora_dropout}\n")
        f.write(f"LoRA target modules: {', '.join(cfg.target_modules)}\n")
        f.write(f"LoRA bias: {cfg.bias}\n\n")

        f.write("=== Parameters ===\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters (LoRA etc.): {trainable_params:,}\n\n")

        f.write("=== Logging / Checkpointing ===\n")
        f.write(f"Output dir: {cfg.output_dir}\n")
        f.write(f"CSV log path: {cfg.csv_log_path}\n")
        f.write(f"Logging steps: {cfg.logging_steps}\n")
        f.write(f"Eval steps: {cfg.eval_steps}\n")
        f.write(f"Save steps: {cfg.save_steps}\n")
        f.write(f"Save total limit: {cfg.save_total_limit}\n")
        f.write(f"Report to: {cfg.report_to}\n")
        f.write(f"Random seed: {SEED}\n\n")

        # ------------------------------------------------------------------
        # Environment and hardware
        # ------------------------------------------------------------------
        import platform
        import transformers

        f.write("=== Environment ===\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"Torch: {torch.__version__}\n")
        f.write(f"Transformers: {transformers.__version__}\n")
        f.write(f"Unsloth: {unsloth.__version__}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA available: True\n")
            f.write(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA capability: {torch.cuda.get_device_capability(0)}\n")
            f.write(f"CUDA version: {torch.version.cuda}\n")
        else:
            f.write("CUDA available: False\n")

    print(f"Run info written to: {run_info_path}")


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    print("Basic setup done. TF32 enabled, seeds set.")

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    print("Config loaded:")
    print(cfg)

    # Hugging Face token (from environment) - used only for model if gated
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("Hugging Face login successful.")
        except Exception as e:
            print(f"HF login failed (continuing, but gated models may not load): {e}")
    else:
        print("HF_TOKEN not set. If base/tokenizer are gated, set HF_TOKEN before running.")

    # Load tokenizer from local dir
    tokenizer = load_tokenizer(cfg)

    # Build datasets
    train_ds, eval_ds = build_train_eval_from_bin(cfg, tokenizer)

    # Quick decode preview
    row0 = eval_ds[0]["input_ids"].tolist()
    preview = tokenizer.decode(row0[:128], skip_special_tokens=False)
    print("\nDecode preview (first 128 tokens of one eval row):")
    print(preview.replace("\n", " ")[:400], "...\n")

    # Load full bf16 base model and attach LoRA
    model_kwargs = {}
    if hf_token:
        model_kwargs["token"] = hf_token

    print("Loading full bf16 base model:", cfg.base_model_id)
    model, _tok_from_model = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model_id,
        max_seq_length=cfg.seq_len,
        dtype=None,          # Unsloth auto-selects bf16 on A100
        load_in_4bit=False,  # No 4-bit quantization
        **model_kwargs,
    )
    print("Full base model loaded (no 4-bit).")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_vocab = getattr(model.config, "vocab_size", None)
    print("Model vocab_size:", model_vocab,
          "| Tokenizer vocab_size:", tokenizer.vocab_size)

    validate_tokenizer_model(tokenizer, _tok_from_model, model)

    print("Attaching LoRA adapters (r=128, alpha=32, dropout=0.0).")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=list(cfg.target_modules),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=cfg.seq_len,
    )
    print("LoRA attached.")

    FastLanguageModel.for_training(model)
    print("Model set to training mode.")

    # Training arguments
    logging_dir = os.path.join(cfg.output_dir, "tb_logs")
    os.makedirs(logging_dir, exist_ok=True)

    training_args = UnslothTrainingArguments(
        output_dir=cfg.output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        embedding_learning_rate=cfg.embedding_learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16 if is_bfloat16_supported() else False,
        fp16=cfg.fp16 and not is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to=cfg.report_to,
        seed=SEED,
        remove_unused_columns=False,
        max_seq_length=cfg.seq_len,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},  # Rows are already packed.
    )

    print("TrainingArguments created.")
    print("num_train_epochs:", training_args.num_train_epochs)
    print("max_steps (should be -1):", training_args.max_steps)
    print("logging_dir:", training_args.logging_dir)

    # Write summary run info (after model + datasets are known)
    write_run_info(cfg, tokenizer, train_ds, eval_ds, model, training_args)

    # Path for generation logging
    generations_path = os.path.join(cfg.output_dir, "generations", "1_generation.txt")

    # Trainer
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fixed_rows,
        args=training_args,
    )

    # The TRL compatibility wrapper can rebuild args as SFTConfig.
    trainer.args.embedding_learning_rate = cfg.embedding_learning_rate

    # Add CSV logger
    trainer.add_callback(CSVLoggerCallback(cfg.csv_log_path))

    # Add generation callback (1 random row from eval after each eval)
    gen_cb = GenerationCallback(
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        generations_path=generations_path,
        max_prompt_tokens=128,
        max_new_tokens=120,
        temperature=0.8,
        top_p=0.9,
    )
    trainer.add_callback(gen_cb)

    print("Trainer initialized. Ready to train.")

    # Training with resume support
    print("\nStarting training (1 epoch). If interrupted, rerun this script to resume.\n")

    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        last_checkpoint = get_last_checkpoint(cfg.output_dir)

    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found, starting fresh training run.")
        train_result = trainer.train()

    print("\nTraining finished.")
    print("Train result:", train_result)

    # Save LoRA adapters and tokenizer
    adapters_dir = "urdu_cpt_lora"
    os.makedirs(adapters_dir, exist_ok=True)
    model.save_pretrained(adapters_dir)
    tokenizer.save_pretrained(adapters_dir)
    print(f"\nLoRA adapters + tokenizer saved to: {adapters_dir}")

    # Final evaluation
    print("\nRunning final evaluation on eval set...")
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    if eval_loss is not None:
        ppl = math.exp(eval_loss)
        print(f"Final eval loss: {eval_loss:.4f} | Perplexity: {ppl:.2f}")
        with open(cfg.csv_log_path, "a", encoding="utf-8") as f:
            f.write(f"{trainer.state.global_step},,{eval_loss},{ppl}\n")
    else:
        print("No eval_loss reported. Check eval set / eval_ratio.")

    # Quick generation sanity check at the end
    from unsloth import FastLanguageModel as _FLM_for_inference

    _FLM_for_inference.for_inference(model)

    prompt = "پاکستان میں آج کی اہم خبروں کی سرخیوں کا خلاصہ پیش کریں:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    print("\nGeneration preview:\n")
    print(tokenizer.decode(gen[0], skip_special_tokens=True))
    print("\nAll done.")


if __name__ == "__main__":
    main()

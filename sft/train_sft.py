"""M3: LoRA SFT on distilled TIR traces (PLAN §7.2).

- Examples come from sft/trace_tokenizer.py (replayed through the real verl agent
  loop): tool-return tokens and the prompt are already -100 in labels.
- LoRA r=32 alpha=64 on all attention+MLP projections; 2 epochs, bf16.
- After training the adapter is merged into a full checkpoint for verl (M4)
  at sft/checkpoints/qwen3-1.7b-sft (model.path-compatible).
- Run artifacts (五件套): resolved config, checkpoint, metrics.json, summary.md
  (predictions come from the M3 acceptance eval, stored under data/probe_sft*).

Long task: run under tmux via scripts/m3/run_sft.sh. DEBUG=1 prints batch shapes.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from sft.trace_tokenizer import MODEL_PATH, trace_to_example  # noqa: E402

TRACES_PATH = os.path.join(PROJECT_DIR, "sft", "data", "sft_traces.jsonl")
OUT_DIR = os.path.join(PROJECT_DIR, "sft", "checkpoints")
DEBUG = os.environ.get("DEBUG", "") == "1"


@dataclass(frozen=True)
class SFTConfig:
    base_model: str = MODEL_PATH
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    epochs: float = 2.0
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    per_device_batch: int = 4
    grad_accum: int = 8  # effective batch 32
    max_len: int = 4096  # prompt (<=1024) + response (<=3072)
    seed: int = 42


class TraceDataset(torch.utils.data.Dataset):
    def __init__(self, traces: list[dict[str, Any]], tokenizer: Any, max_len: int):
        self.examples = []
        n_dropped = 0
        for t in traces:
            ex = trace_to_example(t, tokenizer)
            if len(ex["input_ids"]) > max_len:
                n_dropped += 1
                continue
            self.examples.append(ex)
        print(f"dataset: {len(self.examples)} examples ({n_dropped} dropped over max_len)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def make_collator(pad_token_id: int):
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = max_len - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_token_id] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        if DEBUG:
            shapes = {k: tuple(v.shape) for k, v in tensors.items()}
            n_supervised = int((tensors["labels"] != -100).sum())
            print(f"[debug] batch shapes {shapes}, supervised tokens {n_supervised}")
            assert tensors["input_ids"].shape == tensors["labels"].shape, "shape mismatch"
        return tensors

    return collate


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    cfg = SFTConfig()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "sft_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    traces = [json.loads(line) for line in open(TRACES_PATH, encoding="utf-8")]
    dataset = TraceDataset(traces, tokenizer, cfg.max_len)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules), task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "trainer"),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="no",  # merged checkpoint saved manually below
        report_to=["tensorboard"],
        logging_dir=os.path.join(OUT_DIR, "tensorboard"),
        seed=cfg.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=make_collator(tokenizer.pad_token_id))
    result = trainer.train()

    merged = model.merge_and_unload()
    merged_dir = os.path.join(OUT_DIR, "qwen3-1.7b-sft")
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    logs = [x for x in trainer.state.log_history if "loss" in x]
    metrics = {
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "final_loss": logs[-1]["loss"] if logs else None,
        "first_loss": logs[0]["loss"] if logs else None,
        "n_examples": len(dataset),
        "loss_curve": [(x["step"], x["loss"]) for x in logs],
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(
            f"# SFT run summary\n\nbase: {cfg.base_model}\nexamples: {len(dataset)}\n"
            f"loss: {metrics['first_loss']} -> {metrics['final_loss']}\n"
            f"runtime: {metrics['train_runtime_s']}s\nmerged checkpoint: {merged_dir}\n"
        )
    print(f"merged checkpoint -> {merged_dir}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "loss_curve"}, indent=2))


if __name__ == "__main__":
    main()

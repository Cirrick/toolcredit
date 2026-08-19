"""Single full-parameter SFT control for the scoped M3 follow-up.

This intentionally matches the accepted LoRA run on data, masking, epochs,
effective batch size, sequence length, precision, and seed. The only training
changes are full-parameter optimization and its conventional lower fixed LR.
There is no hyperparameter search.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import torch

from sft.train_sft import TRACES_PATH, TraceDataset, make_collator
from sft.trace_tokenizer import MODEL_PATH

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(PROJECT_DIR, "sft", "experiments", "m3_minimal", "full_sft")


@dataclass(frozen=True)
class FullSFTConfig:
    base_model: str = MODEL_PATH
    traces_path: str = TRACES_PATH
    training_mode: str = "full_parameter"
    epochs: float = 2.0
    lr: float = 1e-5
    warmup_ratio: float = 0.03
    per_device_batch: int = 4
    grad_accum: int = 8
    effective_batch: int = 32
    max_len: int = 4096
    precision: str = "bf16"
    seed: int = 42


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
        raise FileExistsError(f"refusing to overwrite completed checkpoint: {checkpoint_dir}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = FullSFTConfig()
    with open(os.path.join(out_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    with open(cfg.traces_path, encoding="utf-8") as f:
        traces = [json.loads(line) for line in f]
    dataset = TraceDataset(traces, tokenizer, cfg.max_len)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total:
        raise RuntimeError(f"full SFT requires all parameters trainable: {trainable}/{total}")
    print(f"trainable parameters: {trainable:,}/{total:,}")

    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "trainer"),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=["tensorboard"],
        logging_dir=os.path.join(out_dir, "tensorboard"),
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    result = trainer.train()
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    metrics = {
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "first_loss": logs[0]["loss"] if logs else None,
        "final_loss": logs[-1]["loss"] if logs else None,
        "n_examples": len(dataset),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "loss_curve": [(entry["step"], entry["loss"]) for entry in logs],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Full-parameter SFT control\n\n"
            f"- base: `{cfg.base_model}`\n"
            f"- data: `{cfg.traces_path}` ({len(dataset)} examples)\n"
            f"- epochs / effective batch / LR: {cfg.epochs} / {cfg.effective_batch} / {cfg.lr}\n"
            f"- loss: {metrics['first_loss']} -> {metrics['final_loss']}\n"
            f"- runtime: {metrics['train_runtime_s']}s\n"
            f"- checkpoint: `{checkpoint_dir}`\n"
        )
    print(json.dumps({key: value for key, value in metrics.items() if key != "loss_curve"}, indent=2))


if __name__ == "__main__":
    main()

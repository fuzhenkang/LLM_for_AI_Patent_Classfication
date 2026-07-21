"""LLM sequence classification with LoRA, QLoRA, rsLoRA, DoRA, or head-only tuning."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import classification_metrics, fit_label_encoder, get_device, save_label_encoder, set_seed, write_metrics  # noqa: E402
try:
    from .baichuan_sequence_classification import BaichuanForSequenceClassification  # type: ignore  # noqa: E402
    from .glm_sequence_classification import GLMForSequenceClassification  # type: ignore  # noqa: E402
    from .llm_registry import MODEL_CONFIGS, get_llm_config  # type: ignore  # noqa: E402
except ImportError:
    from baichuan_sequence_classification import BaichuanForSequenceClassification  # noqa: E402
    from glm_sequence_classification import GLMForSequenceClassification  # noqa: E402
    from llm_registry import MODEL_CONFIGS, get_llm_config  # noqa: E402


class SequenceClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LLM sequence classifier.")
    parser.add_argument("--model-key", default="qwen", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-csv", required=True, help="Training CSV.")
    parser.add_argument("--valid-csv", required=True, help="Validation CSV for model selection.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--text-cols", default=None, help="Comma-separated input columns, for example: title,abstract,IPC.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=1, help="Number of batches to accumulate before one optimizer update.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--tuning-mode", default="qlora", choices=["lora", "qlora", "rslora", "dora", "head_only"])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target-modules", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true", help="Print step and epoch training logs.")
    parser.add_argument("--save-checkpoint-steps", type=int, default=0, help="Save checkpoint-last every N training steps. Use 0 to disable step checkpoints.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to a checkpoint directory, for example outputs/v1/qwen_qlora/checkpoint-last.")
    args = parser.parse_args()
    return apply_model_defaults(args)


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = get_llm_config(args.model_key)
    if args.base_model is None:
        args.base_model = config.base_model
    if args.lora_target_modules is None:
        args.lora_target_modules = config.lora_target_modules
    if args.max_len is None:
        args.max_len = config.max_len
    if args.batch_size is None:
        args.batch_size = config.batch_size
    if args.lr is None:
        args.lr = config.lr
    if args.torch_dtype is None:
        args.torch_dtype = config.torch_dtype
    args.trust_remote_code = bool(args.trust_remote_code or config.trust_remote_code)
    if args.tuning_mode == "qlora" and not args.load_in_8bit:
        args.load_in_4bit = True
    if args.gradient_steps < 1:
        raise ValueError("--gradient-steps must be >= 1.")
    return args


def parse_text_columns(args: argparse.Namespace) -> list[str]:
    if args.text_cols:
        return [column.strip() for column in args.text_cols.split(",") if column.strip()]
    return [args.text_col]


def build_input_texts(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns: {', '.join(missing)}")
    if len(columns) == 1:
        return df[columns[0]].fillna("").astype(str)
    return df[columns].fillna("").astype(str).agg(" ".join, axis=1)


def read_dataset(path: str, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=args.encoding)
    text_columns = parse_text_columns(args)
    df = df.copy()
    df["_model_text"] = build_input_texts(df, text_columns)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")
    return df


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def patch_tied_weights_keys(model) -> None:
    if not hasattr(model, "all_tied_weights_keys") and hasattr(model, "_tied_weights_keys"):
        model.all_tied_weights_keys = model._tied_weights_keys


def is_baichuan_model(args: argparse.Namespace) -> bool:
    return args.model_key == "baichuan" or "baichuan" in str(args.base_model).lower()


def is_glm_model(args: argparse.Namespace) -> bool:
    base_model = str(args.base_model).lower()
    return args.model_key == "glm" or "chatglm" in base_model or "glm" in base_model


def build_model(args: argparse.Namespace, label_names: list[str]):
    id2label = {idx: str(label) for idx, label in enumerate(label_names)}
    label2id = {str(label): idx for idx, label in enumerate(label_names)}
    model_kwargs = {
        "num_labels": len(label_names),
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": args.trust_remote_code,
    }
    if is_glm_model(args):
        model_kwargs["attn_implementation"] = "eager"
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = dtype_from_name(args.torch_dtype)
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")
    if (is_baichuan_model(args) or is_glm_model(args)) and torch.cuda.is_available():
        model_kwargs.setdefault("device_map", {"": 0})
    if args.load_in_4bit or args.load_in_8bit:
        if is_baichuan_model(args):
            if args.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_quant_type"] = args.bnb_4bit_quant_type
                model_kwargs["bnb_4bit_compute_dtype"] = dtype_from_name(args.bnb_4bit_compute_dtype)
                model_kwargs["bnb_4bit_use_double_quant"] = args.bnb_4bit_use_double_quant
            else:
                model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"
        else:
            from transformers import BitsAndBytesConfig

            if args.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=dtype_from_name(args.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = {"": 0} if is_glm_model(args) and torch.cuda.is_available() else "auto"
    if is_baichuan_model(args):
        model = BaichuanForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)
    elif is_glm_model(args):
        model = GLMForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)
    patch_tied_weights_keys(model)
    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id

    if args.tuning_mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(part in name for part in ("classifier", "score", "classification_head")):
                param.requires_grad = True
        return model

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    lora_kwargs = {
        "task_type": TaskType.SEQ_CLS,
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": parse_target_modules(args.lora_target_modules),
        "bias": "none",
    }
    if args.tuning_mode == "rslora":
        lora_kwargs["use_rslora"] = True
    if args.tuning_mode == "dora":
        lora_kwargs["use_dora"] = True
    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    model.print_trainable_parameters()
    return model


def evaluate(model, loader: DataLoader, device: torch.device, label_names: list[str]) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            model_batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**model_batch)
            losses.append(float(outputs.loss.item()))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist())
    metrics = classification_metrics(y_true, y_pred, label_names)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def save_run_artifacts(args: argparse.Namespace, output_dir: Path, tokenizer, encoder, model, metrics: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    if args.tuning_mode == "head_only":
        torch.save(trainable_state_dict(model), output_dir / "head_state.pt")
    else:
        model.save_pretrained(output_dir / "adapter")
    save_label_encoder(encoder, output_dir)
    config = vars(args).copy()
    config.update(
        {
            "model_type": "llm_sequence_classification",
            "classifier_version": "v1",
            "classification_form": "sequence_classification",
            "best_valid_metrics": metrics,
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def save_checkpoint(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    tokenizer,
    model,
    optimizer,
    scheduler,
    encoder,
    epoch: int,
    step: int,
    global_step: int,
    best_score: float,
    best_metrics: dict[str, object],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    save_label_encoder(encoder, checkpoint_dir)
    torch.save(trainable_state_dict(model), checkpoint_dir / "trainable_model_state.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    state = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "best_score": best_score,
        "best_metrics": best_metrics,
    }
    with (checkpoint_dir / "trainer_state.json").open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)
    config = vars(args).copy()
    config.update(
        {
            "checkpoint_type": "training_resume",
            "model_type": "llm_sequence_classification",
            "classifier_version": "v1",
        }
    )
    with (checkpoint_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(checkpoint_dir: str | Path, model, optimizer, scheduler, device: torch.device) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_dir)
    state_path = checkpoint_path / "trainer_state.json"
    model_state_path = checkpoint_path / "trainable_model_state.pt"
    optimizer_path = checkpoint_path / "optimizer.pt"
    scheduler_path = checkpoint_path / "scheduler.pt"
    required = [state_path, model_state_path, optimizer_path, scheduler_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint files: {', '.join(missing)}")

    model_state = torch.load(model_state_path, map_location="cpu")
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
    move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
    with state_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def next_checkpoint_position(epoch: int, step: int, steps_per_epoch: int) -> tuple[int, int]:
    if step >= steps_per_epoch:
        return epoch + 1, 1
    return epoch, step + 1


def train_once(args: argparse.Namespace, train_df: pd.DataFrame, valid_df: pd.DataFrame, output_dir: Path, seed: int) -> dict[str, object]:
    set_seed(seed)
    device = get_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = fit_label_encoder(train_df[args.label_col], valid_df[args.label_col])
    y_train = encoder.transform(train_df[args.label_col])
    y_valid = encoder.transform(valid_df[args.label_col])
    label_names = [str(label) for label in encoder.classes_]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        SequenceClassificationDataset(train_df["_model_text"], y_train, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        SequenceClassificationDataset(valid_df["_model_text"], y_valid, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(args, label_names)
    if not (args.load_in_4bit or args.load_in_8bit) and not (is_baichuan_model(args) or is_glm_model(args)):
        model.to(device)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_steps))
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_metrics: dict[str, object] = {}
    start_epoch = 1
    start_step = 1
    global_step = 0
    if args.resume_from_checkpoint:
        checkpoint_state = load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint_state.get("epoch", 1))
        start_step = int(checkpoint_state.get("step", 1))
        global_step = int(checkpoint_state.get("global_step", 0))
        best_f1 = float(checkpoint_state.get("best_score", -1.0))
        best_metrics = dict(checkpoint_state.get("best_metrics", {}))
        if args.verbose:
            print(f"resumed from {args.resume_from_checkpoint}: epoch={start_epoch} step={start_step} global_step={global_step}", flush=True)

    checkpoint_last = output_dir / "checkpoint-last"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and step < start_step:
                continue
            model_batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**model_batch).loss
            train_losses.append(float(loss.item()))
            loss = loss / args.gradient_steps
            loss.backward()
            should_update = step % args.gradient_steps == 0 or step == len(train_loader)
            if should_update:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            if args.verbose and (step % 20 == 0 or step == len(train_loader)):
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={sum(train_losses) / len(train_losses):.4f}", flush=True)
            if should_update and args.save_checkpoint_steps > 0 and global_step % args.save_checkpoint_steps == 0:
                next_epoch, next_step = next_checkpoint_position(epoch, step, len(train_loader))
                save_checkpoint(
                    args,
                    checkpoint_last,
                    tokenizer,
                    model,
                    optimizer,
                    scheduler,
                    encoder,
                    next_epoch,
                    next_step,
                    global_step,
                    best_f1,
                    best_metrics,
                )

        metrics = evaluate(model, valid_loader, device, label_names)
        metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        metrics["epoch"] = epoch
        write_metrics(metrics, output_dir / f"valid_metrics_epoch_{epoch}.json")
        if args.verbose:
            print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} valid_f1_macro={metrics['f1_macro']:.4f}", flush=True)
        if float(metrics["f1_macro"]) > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_metrics = metrics
            save_run_artifacts(args, output_dir, tokenizer, encoder, model, metrics)
        save_checkpoint(
            args,
            checkpoint_last,
            tokenizer,
            model,
            optimizer,
            scheduler,
            encoder,
            epoch + 1,
            1,
            global_step,
            best_f1,
            best_metrics,
        )

    write_metrics(best_metrics, output_dir / "best_valid_metrics.json")
    return best_metrics


def train(args: argparse.Namespace) -> dict[str, object]:
    args = apply_model_defaults(args)
    train_df = read_dataset(args.train_csv, args)
    valid_df = read_dataset(args.valid_csv, args)
    return train_once(args, train_df, valid_df, Path(args.output_dir), args.seed)


def main() -> int:
    metrics = train(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

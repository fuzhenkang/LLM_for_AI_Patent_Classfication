"""LLM sequence classification with LoRA, QLoRA, rsLoRA, DoRA, or head-only tuning."""

from __future__ import annotations

import argparse
import json
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
    from .llm_registry import MODEL_CONFIGS, get_llm_config  # type: ignore  # noqa: E402
except ImportError:
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
    parser.add_argument("--data-csv", help="Training CSV for k-fold cross-validation.")
    parser.add_argument("--train-csv", help="Training CSV for final training.")
    parser.add_argument("--valid-csv", help="Validation CSV for model selection.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--text-cols", default=None, help="Comma-separated input columns, for example: title,abstract,IPC.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--fold-col", default="cv_fold")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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


def stratified_folds(labels: pd.Series, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for _, group_indices in labels.groupby(labels).groups.items():
        indices = np.array(list(group_indices))
        rng.shuffle(indices)
        for idx, row_index in enumerate(indices):
            folds[idx % n_splits].append(int(row_index))
    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]


def average_metrics(metrics_list: list[dict[str, object]]) -> dict[str, object]:
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "loss"]
    averaged = {
        key: float(np.mean([float(metrics.get(key, 0.0)) for metrics in metrics_list]))
        for key in keys
        if any(key in metrics for metrics in metrics_list)
    }
    averaged["fold_metrics"] = metrics_list
    return averaged


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_model(args: argparse.Namespace, label_names: list[str]):
    id2label = {idx: str(label) for idx, label in enumerate(label_names)}
    label2id = {str(label): idx for idx, label in enumerate(label_names)}
    model_kwargs = {
        "num_labels": len(label_names),
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = dtype_from_name(args.torch_dtype)
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")
    if args.load_in_4bit or args.load_in_8bit:
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
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)
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
            "best_valid_metrics": metrics,
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


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
    if not (args.load_in_4bit or args.load_in_8bit):
        model.to(device)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_metrics: dict[str, object] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            model_batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**model_batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        metrics = evaluate(model, valid_loader, device, label_names)
        metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        metrics["epoch"] = epoch
        write_metrics(metrics, output_dir / f"valid_metrics_epoch_{epoch}.json")
        print(f"epoch={epoch} train_loss={metrics['train_loss']:.4f} valid_f1_macro={metrics['f1_macro']:.4f}")
        if float(metrics["f1_macro"]) > best_f1:
            best_f1 = float(metrics["f1_macro"])
            best_metrics = metrics
            save_run_artifacts(args, output_dir, tokenizer, encoder, model, metrics)

    write_metrics(best_metrics, output_dir / "best_valid_metrics.json")
    return best_metrics


def cross_validate(args: argparse.Namespace) -> dict[str, object]:
    data_df = read_dataset(str(args.data_csv), args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fold_col in data_df.columns:
        fold_ids = sorted(data_df[args.fold_col].dropna().unique().tolist())
        folds = [data_df.index[data_df[args.fold_col] == fold_id].to_numpy() for fold_id in fold_ids]
    else:
        folds = stratified_folds(data_df[args.label_col], args.cv_folds, args.seed)

    fold_metrics: list[dict[str, object]] = []
    for fold_idx, valid_indices in enumerate(folds):
        valid_set = set(valid_indices.tolist())
        train_indices = [idx for idx in data_df.index if idx not in valid_set]
        train_df = data_df.loc[train_indices].reset_index(drop=True)
        valid_df = data_df.loc[valid_indices].reset_index(drop=True)
        print(f"fold={fold_idx + 1}/{len(folds)} train={len(train_df)} valid={len(valid_df)}")
        metrics = train_once(args, train_df, valid_df, output_dir / f"fold_{fold_idx:02d}", args.seed + fold_idx)
        fold_metrics.append(metrics)

    cv_metrics = average_metrics(fold_metrics)
    write_metrics(cv_metrics, output_dir / "cv_metrics.json")
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args) | {"model_type": "llm_sequence_classification", "cv_folds_actual": len(folds)}, file, ensure_ascii=False, indent=2)
    return cv_metrics


def train(args: argparse.Namespace) -> dict[str, object]:
    args = apply_model_defaults(args)
    if args.data_csv:
        return cross_validate(args)
    if not args.train_csv:
        raise ValueError("Use --data-csv for k-fold cross-validation, or provide --train-csv.")
    train_df = read_dataset(args.train_csv, args)
    if args.valid_csv:
        valid_df = read_dataset(args.valid_csv, args)
        return train_once(args, train_df, valid_df, Path(args.output_dir), args.seed)
    return train_once(args, train_df, train_df, Path(args.output_dir), args.seed)


def main() -> int:
    metrics = train(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

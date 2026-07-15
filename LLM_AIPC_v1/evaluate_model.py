"""Evaluate an LLM sequence-classification model on a held-out test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import classification_metrics, load_label_encoder, write_metrics  # noqa: E402
try:
    from .llm_classifier import SequenceClassificationDataset, build_input_texts, dtype_from_name  # type: ignore  # noqa: E402
except ImportError:
    from llm_classifier import SequenceClassificationDataset, build_input_texts, dtype_from_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LLM sequence classifier.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--text-cols", default=None)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_quantized_kwargs(config: dict[str, object]) -> dict[str, object]:
    kwargs: dict[str, object] = {"trust_remote_code": bool(config.get("trust_remote_code", False))}
    torch_dtype = str(config.get("torch_dtype", "auto"))
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = dtype_from_name(torch_dtype)
    if config.get("load_in_4bit") or config.get("load_in_8bit"):
        from transformers import BitsAndBytesConfig

        if config.get("load_in_4bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
                bnb_4bit_compute_dtype=dtype_from_name(str(config.get("bnb_4bit_compute_dtype", "float16"))),
                bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", False)),
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    return kwargs


def parse_text_columns(args: argparse.Namespace, config: dict[str, object]) -> list[str]:
    if args.text_cols:
        return [column.strip() for column in args.text_cols.split(",") if column.strip()]
    if args.text_col:
        return [args.text_col]
    if config.get("text_cols"):
        return [column.strip() for column in str(config["text_cols"]).split(",") if column.strip()]
    return [str(config.get("text_col", "text"))]


def load_model(model_dir: Path, config: dict[str, object], label_names: list[str], tokenizer):
    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config["base_model"]),
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        **build_quantized_kwargs(config),
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if config.get("tuning_mode") == "head_only":
        state_path = model_dir / "head_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing head-only state file: {state_path}")
        model.load_state_dict(torch.load(state_path, map_location="cpu"), strict=False)
        return model

    from peft import PeftModel

    adapter_dir = model_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")
    return PeftModel.from_pretrained(model, str(adapter_dir))


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", trust_remote_code=bool(config.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = load_label_encoder(model_dir)
    label_names = [str(label) for label in encoder.classes_]
    test_df = pd.read_csv(args.test_csv, encoding=args.encoding)
    texts = build_input_texts(test_df, parse_text_columns(args, config))
    labels = encoder.transform(test_df[args.label_col])
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    loader = DataLoader(SequenceClassificationDataset(texts, labels, tokenizer, int(config["max_len"])), batch_size=batch_size)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir, config, label_names, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")):
        model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels_tensor = batch["labels"].to(device)
            model_batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**model_batch).logits
            y_true.extend(labels_tensor.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, label_names)
    write_metrics(metrics, output_dir / "test_metrics.json")
    predictions = test_df.copy()
    predictions["true_label"] = [label_names[idx] for idx in y_true]
    predictions["pred_label"] = [label_names[idx] for idx in y_pred]
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding=args.encoding)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

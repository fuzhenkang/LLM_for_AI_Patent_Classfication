"""Evaluate a prompt-based next-token classifier on a held-out test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Models.common import classification_metrics, load_label_encoder, write_metrics  # noqa: E402
from PromptClassification.prompt_classifier import PromptClassificationDataset, last_token_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prompt-based next-token classifier.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


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


def load_model(model_dir: Path, config: dict[str, object], tokenizer):
    if config.get("tuning_mode") == "head_only":
        return AutoModelForCausalLM.from_pretrained(model_dir / "model", trust_remote_code=bool(config.get("trust_remote_code", False)))

    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(str(config["base_model"]), **build_quantized_kwargs(config))
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    return PeftModel.from_pretrained(base_model, model_dir / "adapter")


def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    tokenizer.padding_side = "left"

    encoder = load_label_encoder(model_dir)
    test_df = pd.read_csv(args.test_csv, encoding=args.encoding)
    labels = encoder.transform(test_df[args.label_col])
    label_words = list(config["label_words"])
    label_token_ids = [int(item) for item in config["label_token_ids"]]
    batch_size = args.batch_size or int(config.get("batch_size", 1))

    loader = DataLoader(
        PromptClassificationDataset(test_df[args.text_col], labels, tokenizer, int(config["max_len"]), str(config["prompt_template"]), label_words),
        batch_size=batch_size,
        shuffle=False,
    )

    device = get_device(args.device)
    model = load_model(model_dir, config, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")):
        model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels_tensor = batch["labels"].to(device)
            model_batch = {key: value.to(device) for key, value in batch.items()}
            logits = last_token_logits(model, model_batch, label_token_ids)
            y_true.extend(labels_tensor.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    label_names = [str(label) for label in encoder.classes_]
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

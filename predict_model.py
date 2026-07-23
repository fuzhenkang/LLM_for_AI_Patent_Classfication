"""Predict labels for new unlabeled patent data with trained LLM_AIPC models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_label_encoder  # noqa: E402
from evaluate_model import (  # noqa: E402
    ar_candidate_scores,
    build_input_texts,
    get_device,
    is_baichuan_v1_config,
    is_glm_v1_config,
    last_token_logits,
    load_v1_model,
    load_v2_model,
    parse_text_columns,
    sft_last_token_logits,
)


class TextOnlyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len: int):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


class PromptDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len: int, template: str, label_words: list[str]):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.template = template
        self.label_words = label_words

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.template.format(text=self.texts[idx], label_words="/".join(self.label_words))
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


class SFTPromptDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len: int, template: str, label_words: list[str]):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.template = template
        self.label_words = label_words

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self.tokenizer.padding_side = "left"
        prompt = self.template.format(text=self.texts[idx], label_words="/".join(self.label_words))
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max(self.max_len - 1, 1),
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict labels for new unlabeled patent data.")
    parser.add_argument("--model-dir", required=True, help="Training output directory containing config.json.")
    parser.add_argument("--input-csv", required=True, help="New unlabeled CSV/TSV file.")
    parser.add_argument("--output-csv", required=True, help="Prediction output file.")
    parser.add_argument("--text-col", default=None, help="Single input text column. Defaults to training config.")
    parser.add_argument("--text-cols", default=None, help="Comma-separated input columns, for example: title,abstract,IPC.")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--sep", default=",", help="Input delimiter. Use '\\t' for TSV.")
    parser.add_argument("--output-sep", default=",", help="Output delimiter. Use '\\t' for TSV.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def normalize_sep(value: str) -> str:
    return "\t" if value == r"\t" else value


def load_config(model_dir: Path) -> dict[str, object]:
    with (model_dir / "config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def batch_logits_to_predictions(logits: torch.Tensor, label_names: list[str]) -> tuple[list[int], list[str], list[list[float]]]:
    probs = torch.softmax(logits.float(), dim=1).detach().cpu()
    pred_ids = torch.argmax(probs, dim=1).numpy().tolist()
    pred_labels = [label_names[int(idx)] for idx in pred_ids]
    return [int(idx) for idx in pred_ids], pred_labels, probs.numpy().tolist()


def append_prediction_columns(df: pd.DataFrame, pred_ids: list[int], pred_labels: list[str], scores: list[list[float]], label_names: list[str]) -> pd.DataFrame:
    output = df.copy()
    output["pred_id"] = pred_ids
    output["pred_label"] = pred_labels
    output["pred_score"] = [float(max(row)) if row else 0.0 for row in scores]
    for idx, label in enumerate(label_names):
        output[f"score_{label}"] = [float(row[idx]) for row in scores]
    return output


def predict_v1(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, label_names: list[str], texts: pd.Series):
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    loader = DataLoader(TextOnlyDataset(texts, tokenizer, int(config["max_len"])), batch_size=batch_size, shuffle=False)
    device = get_device(args.device)
    model = load_v1_model(model_dir, config, label_names, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")) and not (
        is_baichuan_v1_config(config) or is_glm_v1_config(config)
    ):
        model.to(device)
    model.eval()

    pred_ids: list[int] = []
    pred_labels: list[str] = []
    all_scores: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            model_batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**model_batch).logits
            batch_ids, batch_labels, batch_scores = batch_logits_to_predictions(logits, label_names)
            pred_ids.extend(batch_ids)
            pred_labels.extend(batch_labels)
            all_scores.extend(batch_scores)
    return pred_ids, pred_labels, all_scores


def predict_v2(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, label_names: list[str], texts: pd.Series):
    label_words = list(config["label_words"])
    label_token_ids = [int(item) for item in config["label_token_ids"]]
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    loader = DataLoader(
        PromptDataset(texts, tokenizer, int(config["max_len"]), str(config["template"]), label_words),
        batch_size=batch_size,
        shuffle=False,
    )
    device = get_device(args.device)
    model = load_v2_model(model_dir, config, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")) and not (
        is_baichuan_v1_config(config) or is_glm_v1_config(config)
    ):
        model.to(device)
    model.eval()

    pred_ids: list[int] = []
    pred_labels: list[str] = []
    all_scores: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            model_batch = {key: value.to(device) for key, value in batch.items()}
            logits = last_token_logits(model, model_batch, label_token_ids)
            batch_ids, batch_labels, batch_scores = batch_logits_to_predictions(logits, label_names)
            pred_ids.extend(batch_ids)
            pred_labels.extend(batch_labels)
            all_scores.extend(batch_scores)
    return pred_ids, pred_labels, all_scores


def predict_v3(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, label_names: list[str], texts: pd.Series):
    label_words = list(config["label_words"])
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    device = get_device(args.device)
    model = load_v2_model(model_dir, config, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")) and not (
        is_baichuan_v1_config(config) or is_glm_v1_config(config)
    ):
        model.to(device)
    model.eval()

    reduction = str(config.get("likelihood_reduction", "mean"))
    with torch.no_grad():
        all_scores = [
            ar_candidate_scores(
                model,
                tokenizer,
                texts,
                label_word,
                str(config["template"]),
                int(config["max_len"]),
                device,
                batch_size,
                reduction,
            )
            for label_word in label_words
        ]
    score_tensor = torch.tensor(all_scores).transpose(0, 1)
    return batch_logits_to_predictions(score_tensor, label_names)


def predict_v4(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, label_names: list[str], texts: pd.Series):
    label_words = list(config["label_words"])
    label_token_ids = [int(item) for item in config["label_token_ids"]]
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    loader = DataLoader(
        SFTPromptDataset(texts, tokenizer, int(config["max_len"]), str(config["template"]), label_words),
        batch_size=batch_size,
        shuffle=False,
    )
    device = get_device(args.device)
    model = load_v2_model(model_dir, config, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")) and not (
        is_baichuan_v1_config(config) or is_glm_v1_config(config)
    ):
        model.to(device)
    model.eval()

    pred_ids: list[int] = []
    pred_labels: list[str] = []
    all_scores: list[list[float]] = []
    with torch.no_grad():
        tokenizer.padding_side = "left"
        for batch in loader:
            model_batch = {key: value.to(device) for key, value in batch.items()}
            logits = sft_last_token_logits(model, model_batch, label_token_ids)
            batch_ids, batch_labels, batch_scores = batch_logits_to_predictions(logits, label_names)
            pred_ids.extend(batch_ids)
            pred_labels.extend(batch_labels)
            all_scores.extend(batch_scores)
    return pred_ids, pred_labels, all_scores


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    config = load_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", trust_remote_code=bool(config.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    encoder = load_label_encoder(model_dir)
    label_names = [str(label) for label in encoder.classes_]
    input_df = pd.read_csv(args.input_csv, encoding=args.encoding, sep=normalize_sep(args.sep))
    texts = build_input_texts(input_df, parse_text_columns(args, config))

    model_type = str(config.get("model_type", ""))
    if model_type == "llm_sequence_classification":
        pred_ids, pred_labels, scores = predict_v1(args, model_dir, config, tokenizer, label_names, texts)
    elif model_type == "llm_next_token_classifier":
        pred_ids, pred_labels, scores = predict_v2(args, model_dir, config, tokenizer, label_names, texts)
    elif model_type == "llm_ar_classifier":
        pred_ids, pred_labels, scores = predict_v3(args, model_dir, config, tokenizer, label_names, texts)
    elif model_type in {"llm_sft_classifier", "llm_ar_pseudo_classifier"}:
        pred_ids, pred_labels, scores = predict_v4(args, model_dir, config, tokenizer, label_names, texts)
    else:
        raise ValueError(f"Unknown model_type in config.json: {model_type}")

    output_df = append_prediction_columns(input_df, pred_ids, pred_labels, scores, label_names)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding=args.encoding, sep=normalize_sep(args.output_sep))
    print(json.dumps({"input_rows": len(input_df), "output_csv": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



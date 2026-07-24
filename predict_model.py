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
    parser.add_argument("--chunk-size", type=int, default=0, help="Rows to predict and save per chunk. Use 0 to predict all rows at once.")
    parser.add_argument("--resume", action="store_true", help="Resume prediction by skipping rows already written to --output-csv.")
    parser.add_argument("--id-col", default=None, help="Stable ID column for resume, for example PN. If omitted, resume skips by completed row count.")
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


def predict_last_token_logits(model, batch: dict[str, torch.Tensor], label_token_ids: list[int]) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    attention_mask = batch["attention_mask"]
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    last_indices = (attention_mask * positions).max(dim=1).values
    batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
    logits = outputs.logits[batch_indices, last_indices]
    return logits[:, torch.tensor(label_token_ids, device=logits.device)]


def should_move_model_to_device(config: dict[str, object], model_type: str) -> bool:
    if config.get("load_in_4bit") or config.get("load_in_8bit"):
        return False
    if model_type == "llm_sequence_classification":
        return not (is_baichuan_v1_config(config) or is_glm_v1_config(config))
    return not is_baichuan_v1_config(config)


def load_prediction_model(
    args: argparse.Namespace,
    model_dir: Path,
    config: dict[str, object],
    tokenizer,
    label_names: list[str],
    model_type: str,
):
    device = get_device(args.device)
    if model_type == "llm_sequence_classification":
        model = load_v1_model(model_dir, config, label_names, tokenizer)
    elif model_type in {"llm_next_token_classifier", "llm_ar_classifier", "llm_sft_classifier", "llm_ar_pseudo_classifier"}:
        model = load_v2_model(model_dir, config, tokenizer)
    else:
        raise ValueError(f"Unknown model_type in config.json: {model_type}")
    if should_move_model_to_device(config, model_type):
        model.to(device)
    model.eval()
    return model, device


def predict_texts_with_model(
    args: argparse.Namespace,
    config: dict[str, object],
    tokenizer,
    label_names: list[str],
    texts: pd.Series,
    model,
    device: torch.device,
    model_type: str,
):
    if model_type == "llm_sequence_classification":
        batch_size = args.batch_size or int(config.get("batch_size", 1))
        loader = DataLoader(TextOnlyDataset(texts, tokenizer, int(config["max_len"])), batch_size=batch_size, shuffle=False)
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

    if model_type == "llm_next_token_classifier":
        label_words = list(config["label_words"])
        label_token_ids = [int(item) for item in config["label_token_ids"]]
        batch_size = args.batch_size or int(config.get("batch_size", 1))
        loader = DataLoader(
            PromptDataset(texts, tokenizer, int(config["max_len"]), str(config["template"]), label_words),
            batch_size=batch_size,
            shuffle=False,
        )
        pred_ids: list[int] = []
        pred_labels: list[str] = []
        all_scores: list[list[float]] = []
        with torch.no_grad():
            for batch in loader:
                model_batch = {key: value.to(device) for key, value in batch.items()}
                logits = predict_last_token_logits(model, model_batch, label_token_ids)
                batch_ids, batch_labels, batch_scores = batch_logits_to_predictions(logits, label_names)
                pred_ids.extend(batch_ids)
                pred_labels.extend(batch_labels)
                all_scores.extend(batch_scores)
        return pred_ids, pred_labels, all_scores

    if model_type == "llm_ar_classifier":
        label_words = list(config["label_words"])
        batch_size = args.batch_size or int(config.get("batch_size", 1))
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

    if model_type in {"llm_sft_classifier", "llm_ar_pseudo_classifier"}:
        label_words = list(config["label_words"])
        label_token_ids = [int(item) for item in config["label_token_ids"]]
        batch_size = args.batch_size or int(config.get("batch_size", 1))
        loader = DataLoader(
            SFTPromptDataset(texts, tokenizer, int(config["max_len"]), str(config["template"]), label_words),
            batch_size=batch_size,
            shuffle=False,
        )
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

    raise ValueError(f"Unknown model_type in config.json: {model_type}")


def predict_dataframe(
    args: argparse.Namespace,
    config: dict[str, object],
    tokenizer,
    label_names: list[str],
    input_df: pd.DataFrame,
    model,
    device: torch.device,
    model_type: str,
) -> pd.DataFrame:
    texts = build_input_texts(input_df, parse_text_columns(args, config))
    pred_ids, pred_labels, scores = predict_texts_with_model(
        args, config, tokenizer, label_names, texts, model, device, model_type
    )
    return append_prediction_columns(input_df, pred_ids, pred_labels, scores, label_names)


def load_resume_state(args: argparse.Namespace, output_path: Path, output_sep: str) -> tuple[set[str], int]:
    if not args.resume or not output_path.exists():
        return set(), 0
    done_df = pd.read_csv(output_path, encoding=args.encoding, sep=output_sep, usecols=[args.id_col] if args.id_col else None)
    if args.id_col:
        if args.id_col not in done_df.columns:
            raise ValueError(f"--id-col {args.id_col} is not present in existing output file: {output_path}")
        return set(done_df[args.id_col].dropna().astype(str)), len(done_df)
    return set(), len(done_df)


def filter_completed_rows(df: pd.DataFrame, args: argparse.Namespace, completed_ids: set[str], completed_rows: int, seen_rows: int) -> pd.DataFrame:
    if args.resume and args.id_col:
        if args.id_col not in df.columns:
            raise ValueError(f"--id-col {args.id_col} is not present in input data.")
        return df[~df[args.id_col].astype(str).isin(completed_ids)].reset_index(drop=True)
    if args.resume and completed_rows > seen_rows:
        return df.iloc[min(completed_rows - seen_rows, len(df)) :].reset_index(drop=True)
    return df


def write_prediction_chunk(output_df: pd.DataFrame, output_path: Path, encoding: str, output_sep: str, append: bool) -> None:
    output_df.to_csv(output_path, index=False, encoding=encoding, sep=output_sep, mode="a" if append else "w", header=not append)


def main() -> int:
    args = parse_args()
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0.")

    model_dir = Path(args.model_dir)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_sep = normalize_sep(args.sep)
    output_sep = normalize_sep(args.output_sep)

    config = load_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", trust_remote_code=bool(config.get("trust_remote_code", False)))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    encoder = load_label_encoder(model_dir)
    label_names = [str(label) for label in encoder.classes_]
    model_type = str(config.get("model_type", ""))
    model, device = load_prediction_model(args, model_dir, config, tokenizer, label_names, model_type)

    completed_ids, completed_rows = load_resume_state(args, output_path, output_sep)
    if not args.resume and output_path.exists():
        output_path.unlink()

    total_input_rows = 0
    predicted_rows = 0
    skipped_rows = 0
    append = args.resume and output_path.exists()

    if args.chunk_size:
        reader = pd.read_csv(args.input_csv, encoding=args.encoding, sep=input_sep, chunksize=args.chunk_size)
        seen_rows = 0
        for chunk_idx, chunk_df in enumerate(reader, start=1):
            original_len = len(chunk_df)
            total_input_rows += original_len
            chunk_df = filter_completed_rows(chunk_df, args, completed_ids, completed_rows, seen_rows)
            seen_rows += original_len
            skipped_rows += original_len - len(chunk_df)
            if chunk_df.empty:
                print(f"chunk={chunk_idx} skipped={original_len} predicted=0")
                continue
            output_df = predict_dataframe(args, config, tokenizer, label_names, chunk_df, model, device, model_type)
            write_prediction_chunk(output_df, output_path, args.encoding, output_sep, append)
            append = True
            predicted_rows += len(output_df)
            print(f"chunk={chunk_idx} skipped={original_len - len(chunk_df)} predicted={len(output_df)} total_predicted={predicted_rows}")
    else:
        input_df = pd.read_csv(args.input_csv, encoding=args.encoding, sep=input_sep)
        total_input_rows = len(input_df)
        input_df = filter_completed_rows(input_df, args, completed_ids, completed_rows, 0)
        skipped_rows = total_input_rows - len(input_df)
        if not input_df.empty:
            output_df = predict_dataframe(args, config, tokenizer, label_names, input_df, model, device, model_type)
            write_prediction_chunk(output_df, output_path, args.encoding, output_sep, append)
            predicted_rows = len(output_df)

    print(
        json.dumps(
            {
                "input_rows": total_input_rows,
                "skipped_rows": skipped_rows,
                "predicted_rows": predicted_rows,
                "resume": bool(args.resume),
                "output_csv": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

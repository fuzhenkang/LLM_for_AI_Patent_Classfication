"""Evaluate LLM_AIPC_v1, LLM_AIPC_v2, LLM_AIPC_v3, or LLM_AIPC_v4 models on a held-out test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import classification_metrics, load_label_encoder, write_metrics  # noqa: E402
from LLM_AIPC_v1.llm_classifier import SequenceClassificationDataset  # noqa: E402
from LLM_AIPC_v1.llm_classifier import dtype_from_name as dtype_from_name_v1  # noqa: E402
from LLM_AIPC_v2.llm_classifier import LLMClassificationDataset, last_token_logits, rebuild_baichuan_rotary_cache  # noqa: E402
from LLM_AIPC_v3.llm_classifier import candidate_scores as ar_candidate_scores  # noqa: E402
from LLM_AIPC_v4.llm_classifier import GPTPseudoClassificationDataset, last_token_logits as pseudo_last_token_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LLM patent classifier.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default=None, help="Single input text column. Defaults to training config.")
    parser.add_argument("--text-cols", default=None, help="Comma-separated input columns. Defaults to training config.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_quantized_kwargs(config: dict[str, object], dtype_fn=dtype_from_name) -> dict[str, object]:
    kwargs: dict[str, object] = {"trust_remote_code": bool(config.get("trust_remote_code", False))}
    torch_dtype = str(config.get("torch_dtype", "auto"))
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = dtype_fn(torch_dtype)
    if config.get("load_in_4bit") or config.get("load_in_8bit"):
        if config.get("use_legacy_bnb_args"):
            kwargs["load_in_4bit"] = bool(config.get("load_in_4bit"))
            kwargs["load_in_8bit"] = bool(config.get("load_in_8bit"))
            if config.get("load_in_4bit"):
                kwargs["bnb_4bit_quant_type"] = str(config.get("bnb_4bit_quant_type", "nf4"))
                kwargs["bnb_4bit_compute_dtype"] = dtype_fn(str(config.get("bnb_4bit_compute_dtype", "float16")))
                kwargs["bnb_4bit_use_double_quant"] = bool(config.get("bnb_4bit_use_double_quant", False))
        else:
            from transformers import BitsAndBytesConfig

            if config.get("load_in_4bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
                    bnb_4bit_compute_dtype=dtype_fn(str(config.get("bnb_4bit_compute_dtype", "float16"))),
                    bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", False)),
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        device_map = str(config.get("device_map", "auto"))
        if device_map == "cuda":
            kwargs["device_map"] = {"": 0}
        elif device_map == "cpu":
            kwargs["device_map"] = {"": "cpu"}
        elif device_map != "none":
            kwargs["device_map"] = device_map
    return kwargs


def load_v2_base_model(base_model: str, config: dict[str, object], **kwargs):
    if config.get("model_loader") == "mistral3_conditional":
        try:
            from transformers import Mistral3ForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "This model needs Mistral3ForConditionalGeneration. Please upgrade transformers, "
                "for example: pip install -U transformers accelerate mistral-common"
            ) from exc

        return Mistral3ForConditionalGeneration.from_pretrained(base_model, **kwargs)
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def resolve_model_subdir(model_dir: Path, subdir: str) -> str:
    path = (model_dir / subdir).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {subdir} directory: {path}. "
            f"Please set --model-dir to the training output directory that contains config.json, tokenizer/, and {subdir}/."
        )
    return str(path)


def adapter_vocab_size(adapter_dir: str) -> int | None:
    adapter_path = Path(adapter_dir)
    safetensors_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"
    candidate_suffixes = ("embed_tokens.weight", "lm_head.weight")

    if safetensors_path.exists():
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt", device="cpu") as file:
            for key in file.keys():
                if key.endswith(candidate_suffixes):
                    return int(file.get_tensor(key).shape[0])

    if bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
        for key, value in state_dict.items():
            if key.endswith(candidate_suffixes):
                return int(value.shape[0])
    return None


def align_model_vocab(model, tokenizer, adapter_dir: str) -> None:
    target_size = adapter_vocab_size(adapter_dir) or len(tokenizer)
    if model.get_input_embeddings().weight.shape[0] != target_size:
        model.resize_token_embeddings(target_size)


def load_v2_model(model_dir: Path, config: dict[str, object], tokenizer):
    if config.get("tuning_mode") == "head_only":
        return load_v2_base_model(resolve_model_subdir(model_dir, "model"), config, trust_remote_code=bool(config.get("trust_remote_code", False)))

    from peft import PeftModel

    base_model = load_v2_base_model(str(config["base_model"]), config, **build_quantized_kwargs(config))
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    adapter_dir = resolve_model_subdir(model_dir, "adapter")
    align_model_vocab(base_model, tokenizer, adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if "baichuan" in str(config.get("base_model", "")).lower():
        rebuild_baichuan_rotary_cache(model)
    return model


def load_v1_model(model_dir: Path, config: dict[str, object], label_names: list[str], tokenizer):
    id2label = {idx: label for idx, label in enumerate(label_names)}
    label2id = {label: idx for idx, label in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config["base_model"]),
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        **build_quantized_kwargs(config, dtype_from_name_v1),
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
    return PeftModel.from_pretrained(model, str(adapter_dir.resolve()))


def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_text_columns(args: argparse.Namespace, config: dict[str, object]) -> list[str]:
    if args.text_cols:
        return [column.strip() for column in args.text_cols.split(",") if column.strip()]
    if args.text_col:
        return [args.text_col]
    if config.get("text_cols"):
        return [column.strip() for column in str(config["text_cols"]).split(",") if column.strip()]
    return [str(config.get("text_col", "text"))]


def build_input_texts(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns: {', '.join(missing)}")
    if len(columns) == 1:
        return df[columns[0]].fillna("").astype(str)
    return df[columns].fillna("").astype(str).agg(" ".join, axis=1)


def write_predictions(test_df: pd.DataFrame, y_true: list[int], y_pred: list[int], label_names: list[str], output_dir: Path, encoding: str) -> None:
    predictions = test_df.copy()
    predictions["true_label"] = [label_names[idx] for idx in y_true]
    predictions["pred_label"] = [label_names[idx] for idx in y_pred]
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding=encoding)


def evaluate_v1(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, encoder, test_df: pd.DataFrame, texts: pd.Series) -> dict[str, object]:
    label_names = [str(label) for label in encoder.classes_]
    labels = encoder.transform(test_df[args.label_col])
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    loader = DataLoader(SequenceClassificationDataset(texts, labels, tokenizer, int(config["max_len"])), batch_size=batch_size)

    device = get_device(args.device)
    model = load_v1_model(model_dir, config, label_names, tokenizer)
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
    write_predictions(test_df, y_true, y_pred, label_names, Path(args.output_dir), args.encoding)
    return metrics


def evaluate_v2(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, encoder, test_df: pd.DataFrame, texts: pd.Series) -> dict[str, object]:
    label_words = list(config["label_words"])
    label_token_ids = [int(item) for item in config["label_token_ids"]]
    labels = encoder.transform(test_df[args.label_col])
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    loader = DataLoader(
        LLMClassificationDataset(texts, labels, tokenizer, int(config["max_len"]), str(config["template"]), label_words),
        batch_size=batch_size,
        shuffle=False,
    )

    device = get_device(args.device)
    model = load_v2_model(model_dir, config, tokenizer)
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
    write_predictions(test_df, y_true, y_pred, label_names, Path(args.output_dir), args.encoding)
    return metrics


def evaluate_v3(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, encoder, test_df: pd.DataFrame, texts: pd.Series) -> dict[str, object]:
    label_words = list(config["label_words"])
    labels = encoder.transform(test_df[args.label_col])
    batch_size = args.batch_size or int(config.get("batch_size", 1))

    device = get_device(args.device)
    model = load_v2_model(model_dir, config, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")):
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
    y_true = labels.tolist()
    y_pred = torch.argmax(score_tensor, dim=1).cpu().numpy().tolist()
    label_names = [str(label) for label in encoder.classes_]
    true_scores = [float(score_tensor[idx, label]) for idx, label in enumerate(y_true)]
    metrics = classification_metrics(y_true, y_pred, label_names)
    metrics["loss"] = float(-sum(true_scores) / max(len(true_scores), 1))
    write_predictions(test_df, y_true, y_pred, label_names, Path(args.output_dir), args.encoding)
    return metrics


def evaluate_v4(args: argparse.Namespace, model_dir: Path, config: dict[str, object], tokenizer, encoder, test_df: pd.DataFrame, texts: pd.Series) -> dict[str, object]:
    label_words = list(config["label_words"])
    label_token_ids = [int(item) for item in config["label_token_ids"]]
    labels = encoder.transform(test_df[args.label_col])
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    loader = DataLoader(
        GPTPseudoClassificationDataset(
            texts,
            labels,
            tokenizer,
            int(config["max_len"]),
            str(config["template"]),
            label_words,
            label_token_ids,
            is_train=False,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    device = get_device(args.device)
    model = load_v2_model(model_dir, config, tokenizer)
    if not (config.get("load_in_4bit") or config.get("load_in_8bit")):
        model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        tokenizer.padding_side = "left"
        for batch in loader:
            labels_tensor = batch["class_labels"].to(device)
            model_batch = {key: value.to(device) for key, value in batch.items() if key != "class_labels"}
            logits = pseudo_last_token_logits(model, model_batch, label_token_ids)
            y_true.extend(labels_tensor.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    label_names = [str(label) for label in encoder.classes_]
    metrics = classification_metrics(y_true, y_pred, label_names)
    write_predictions(test_df, y_true, y_pred, label_names, Path(args.output_dir), args.encoding)
    return metrics


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
    texts = build_input_texts(test_df, parse_text_columns(args, config))
    model_type = str(config.get("model_type", ""))
    if model_type == "llm_sequence_classification":
        metrics = evaluate_v1(args, model_dir, config, tokenizer, encoder, test_df, texts)
    elif model_type == "llm_next_token_classifier":
        metrics = evaluate_v2(args, model_dir, config, tokenizer, encoder, test_df, texts)
    elif model_type == "llm_ar_classifier":
        metrics = evaluate_v3(args, model_dir, config, tokenizer, encoder, test_df, texts)
    elif model_type == "llm_ar_pseudo_classifier":
        metrics = evaluate_v4(args, model_dir, config, tokenizer, encoder, test_df, texts)
    else:
        raise ValueError(f"Unknown model_type in config.json: {model_type}")

    write_metrics(metrics, output_dir / "test_metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

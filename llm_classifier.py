"""LLM next-token classification with LoRA, QLoRA, rsLoRA, DoRA, or head-only tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import classification_metrics, fit_label_encoder, get_device, save_label_encoder, set_seed, write_metrics  # noqa: E402
from llm_registry import MODEL_CONFIGS, get_llm_model_config  # noqa: E402


DEFAULT_TEMPLATE = "请判断以下专利是否属于人工智能专利。只回答“{label_words}”中的一个。\n专利文本：{text}\n答案："


class LLMClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int, template: str, label_words: list[str]):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.template = template
        self.label_words = label_words

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.template.format(text=self.texts[idx], label_words="/".join(self.label_words))
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LLM next-token classifier.")
    parser.add_argument("--model-key", default="qwen", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument(
        "--text-cols",
        default=None,
        help="Comma-separated input columns to concatenate, for example: title,abstract,IPC. Overrides --text-col.",
    )
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--label-words", default="否,是", help="Comma-separated verbalizer words ordered by encoded label class.")
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
    parser.add_argument("--use-legacy-bnb-args", action="store_true", help="Use legacy load_in_4bit/load_in_8bit kwargs instead of BitsAndBytesConfig.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true", help="Print step and epoch training logs.")
    parser.add_argument("--save-checkpoint-steps", type=int, default=0, help="Save checkpoint-last every N training steps. Use 0 to disable step checkpoints.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to a checkpoint directory, for example outputs/llm/qwen_qlora/checkpoint-last.")
    args = parser.parse_args()
    return apply_model_defaults(args)


def apply_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = get_llm_model_config(args.model_key)
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
    args.use_legacy_bnb_args = bool(args.use_legacy_bnb_args or config.use_legacy_bnb_args)
    if args.tuning_mode == "qlora" and not args.load_in_8bit:
        args.load_in_4bit = True
    return args


def parse_text_columns(args: argparse.Namespace) -> list[str]:
    if getattr(args, "text_cols", None):
        return [column.strip() for column in args.text_cols.split(",") if column.strip()]
    return [args.text_col]


def build_input_texts(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns: {', '.join(missing)}")
    if len(columns) == 1:
        return df[columns[0]].fillna("").astype(str)
    return df[columns].fillna("").astype(str).agg(" ".join, axis=1)


def parse_target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_label_token_ids(tokenizer, label_words: list[str]) -> list[int]:
    token_ids: list[int] = []
    for word in label_words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Label word cannot be tokenized: {word}")
        token_ids.append(int(encoded[0]))
    return token_ids


def build_model(args: argparse.Namespace, tokenizer):
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")

    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = dtype_from_name(args.torch_dtype)

    if args.load_in_4bit or args.load_in_8bit:
        if args.use_legacy_bnb_args:
            model_kwargs["load_in_4bit"] = bool(args.load_in_4bit)
            model_kwargs["load_in_8bit"] = bool(args.load_in_8bit)
            if args.load_in_4bit:
                model_kwargs["bnb_4bit_quant_type"] = args.bnb_4bit_quant_type
                model_kwargs["bnb_4bit_compute_dtype"] = dtype_from_name(args.bnb_4bit_compute_dtype)
                model_kwargs["bnb_4bit_use_double_quant"] = args.bnb_4bit_use_double_quant
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
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if args.tuning_mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lm_head" in name or "embed_out" in name or "output_layer" in name:
                param.requires_grad = True
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        if trainable == 0:
            raise ValueError("No output head parameters were found for head_only tuning.")
        if args.verbose:
            print(f"trainable head params: {trainable}")
        return model

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    lora_kwargs = {
        "task_type": TaskType.CAUSAL_LM,
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
    if args.verbose:
        model.print_trainable_parameters()
    return model


def last_token_logits(model, batch: dict[str, torch.Tensor], label_token_ids: list[int]) -> torch.Tensor:
    labels = batch.pop("labels")
    outputs = model(**batch)
    attention_mask = batch["attention_mask"]
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    last_indices = (attention_mask * positions).max(dim=1).values
    batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
    logits = outputs.logits[batch_indices, last_indices]
    selected = logits[:, torch.tensor(label_token_ids, device=logits.device)]
    batch["labels"] = labels
    return selected


def evaluate(model, loader: DataLoader, device: torch.device, label_words: list[str], label_token_ids: list[int]) -> dict[str, object]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            model_batch = {key: value.to(device) for key, value in batch.items()}
            logits = last_token_logits(model, model_batch, label_token_ids)
            loss = loss_fn(logits, labels)
            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    metrics = classification_metrics(y_true, y_pred, label_words)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def save_best_model(args: argparse.Namespace, output_dir: Path, tokenizer, model, encoder, label_words: list[str], label_token_ids: list[int], metrics: dict[str, object]) -> None:
    tokenizer.save_pretrained(output_dir / "tokenizer")
    if args.tuning_mode == "head_only":
        model.save_pretrained(output_dir / "model")
    else:
        model.save_pretrained(output_dir / "adapter")
    save_label_encoder(encoder, output_dir)
    config = vars(args).copy()
    config.update(
        {
            "model_type": "llm_next_token_classifier",
            "label_words": label_words,
            "label_token_ids": label_token_ids,
            "best_valid_metrics": metrics,
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


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
    config["checkpoint_type"] = "training_resume"
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


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv, encoding=args.encoding)
    valid_df = pd.read_csv(args.valid_csv, encoding=args.encoding)
    text_columns = parse_text_columns(args)
    train_texts = build_input_texts(train_df, text_columns)
    valid_texts = build_input_texts(valid_df, text_columns)
    encoder = fit_label_encoder(train_df[args.label_col], valid_df[args.label_col])
    y_train = encoder.transform(train_df[args.label_col])
    y_valid = encoder.transform(valid_df[args.label_col])
    label_words = [item.strip() for item in args.label_words.split(",") if item.strip()]
    if len(label_words) != len(encoder.classes_):
        raise ValueError(f"--label-words has {len(label_words)} words, but data has {len(encoder.classes_)} classes.")

    device = get_device(args.device)
    tokenizer = build_tokenizer(args)
    label_token_ids = build_label_token_ids(tokenizer, label_words)
    model = build_model(args, tokenizer)
    if not (args.load_in_4bit or args.load_in_8bit):
        model.to(device)

    train_loader = DataLoader(
        LLMClassificationDataset(train_texts, y_train, tokenizer, args.max_len, args.template, label_words),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        LLMClassificationDataset(valid_texts, y_valid, tokenizer, args.max_len, args.template, label_words),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_score = -1.0
    best_metrics: dict[str, object] = {}
    start_epoch = 1
    start_step = 1
    global_step = 0
    if args.resume_from_checkpoint:
        checkpoint_state = load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint_state.get("epoch", 1))
        start_step = int(checkpoint_state.get("step", 1))
        global_step = int(checkpoint_state.get("global_step", 0))
        best_score = float(checkpoint_state.get("best_score", -1.0))
        best_metrics = dict(checkpoint_state.get("best_metrics", {}))
        if args.verbose:
            print(f"resumed from {args.resume_from_checkpoint}: epoch={start_epoch} step={start_step} global_step={global_step}", flush=True)

    checkpoint_last = output_dir / "checkpoint-last"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and step < start_step:
                continue
            labels = batch["labels"].to(device)
            model_batch = {key: value.to(device) for key, value in batch.items()}
            logits = last_token_logits(model, model_batch, label_token_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            losses.append(float(loss.item()))
            if args.verbose and (step % 20 == 0 or step == len(train_loader)):
                print(f"epoch={epoch} step={step}/{len(train_loader)} train_loss={sum(losses) / len(losses):.4f}", flush=True)
            if args.save_checkpoint_steps > 0 and global_step % args.save_checkpoint_steps == 0:
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
                    best_score,
                    best_metrics,
                )

        metrics = evaluate(model, valid_loader, device, label_words, label_token_ids)
        metrics["train_loss"] = float(sum(losses) / max(len(losses), 1))
        metrics["epoch"] = epoch
        write_metrics(metrics, output_dir / f"valid_metrics_epoch_{epoch}.json")
        if args.verbose:
            print(f"epoch={epoch} valid_loss={metrics['loss']:.4f} f1_macro={metrics['f1_macro']:.4f}", flush=True)

        score = float(metrics.get("f1_macro", 0.0))
        if score > best_score:
            best_score = score
            best_metrics = metrics
            save_best_model(args, output_dir, tokenizer, model, encoder, label_words, label_token_ids, metrics)
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
            best_score,
            best_metrics,
        )

    write_metrics(best_metrics, output_dir / "best_valid_metrics.json")
    return best_metrics


def main() -> int:
    metrics = train(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

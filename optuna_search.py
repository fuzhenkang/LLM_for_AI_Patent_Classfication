"""Common Optuna search for LLM_AIPC_v1 and LLM_AIPC_v2."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from LLM_AIPC_v1.llm_classifier import apply_model_defaults as apply_v1_defaults  # noqa: E402
from LLM_AIPC_v1.llm_classifier import train as train_v1  # noqa: E402
from LLM_AIPC_v1.llm_registry import MODEL_CONFIGS as V1_MODEL_CONFIGS  # noqa: E402
from LLM_AIPC_v2.llm_classifier import apply_model_defaults as apply_v2_defaults  # noqa: E402
from LLM_AIPC_v2.llm_classifier import train as train_v2  # noqa: E402
from LLM_AIPC_v2.llm_registry import MODEL_CONFIGS as V2_MODEL_CONFIGS  # noqa: E402


MODEL_KEYS = sorted(set(V1_MODEL_CONFIGS) | set(V2_MODEL_CONFIGS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune LLM patent classifier hyperparameters on a validation set.")
    parser.add_argument("--classifier-version", default="v2", choices=["v1", "v2"], help="v1 uses a sequence classification head; v2 uses next-token prediction.")
    parser.add_argument("--model-key", default="qwen", choices=MODEL_KEYS)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--text-cols", default=None, help="Comma-separated input columns, for example: title,abstract,IPC.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--template", default=None, help="Only used by v2 next-token classification.")
    parser.add_argument("--label-words", default=None, help="Only used by v2. Comma-separated verbalizer words ordered by encoded label class.")
    parser.add_argument("--tuning-mode", default="qlora", choices=["lora", "qlora", "rslora", "dora", "head_only"])
    parser.add_argument("--lora-target-modules", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--use-legacy-bnb-args", action="store_true")
    parser.add_argument("--device-map", default=None, choices=["auto", "cuda", "cpu", "none"])
    parser.add_argument("--model-loader", default=None, choices=["causal_lm", "mistral3_conditional"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--metric", default="f1_macro", choices=["f1_macro", "accuracy", "precision_macro", "recall_macro"])
    return parser.parse_args()


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.classifier_version == "v1":
        if args.model_key not in V1_MODEL_CONFIGS:
            raise ValueError(f"model-key {args.model_key} is not supported by LLM_AIPC_v1.")
        return apply_v1_defaults(args)
    if args.model_key not in V2_MODEL_CONFIGS:
        raise ValueError(f"model-key {args.model_key} is not supported by LLM_AIPC_v2.")
    args = apply_v2_defaults(args)
    if args.template is None:
        from LLM_AIPC_v2.llm_classifier import DEFAULT_TEMPLATE

        args.template = DEFAULT_TEMPLATE
    if args.label_words is None:
        args.label_words = "否,是"
    return args


def run_train(args: argparse.Namespace) -> dict[str, object]:
    if args.classifier_version == "v1":
        return train_v1(args)
    return train_v2(args)


def suggest_args(base_args: argparse.Namespace, trial: optuna.Trial) -> argparse.Namespace:
    trial_args = argparse.Namespace(**vars(base_args))
    trial_args.output_dir = str(Path(base_args.output_dir) / f"trial_{trial.number:04d}")
    trial_args.lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    trial_args.max_len = trial.suggest_categorical("max_len", [128, 256, 384])
    trial_args.batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    trial_args.weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    trial_args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    trial_args.save_checkpoint_steps = 0
    trial_args.resume_from_checkpoint = None
    if trial_args.tuning_mode != "head_only":
        trial_args.lora_r = trial.suggest_categorical("lora_r", [4, 8, 16])
        trial_args.lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32, 64])
        trial_args.lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
    else:
        trial_args.lora_r = 0
        trial_args.lora_alpha = 0
        trial_args.lora_dropout = 0.0
    return apply_defaults(trial_args)


def main() -> int:
    args = apply_defaults(parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        trial_args = suggest_args(args, trial)
        metrics = run_train(trial_args)
        score = float(metrics.get(args.metric, 0.0))
        with (Path(trial_args.output_dir) / "trial_params.json").open("w", encoding="utf-8") as file:
            json.dump(vars(trial_args), file, ensure_ascii=False, indent=2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best_model_dir = output_dir / f"trial_{study.best_trial.number:04d}"
    final_dir = output_dir / "best_model"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    if best_model_dir.exists():
        shutil.copytree(best_model_dir, final_dir)

    best_params = dict(study.best_trial.params)
    best_params.update(
        {
            "classifier_version": args.classifier_version,
            "model_key": args.model_key,
            "base_model": args.base_model,
            "tuning_mode": args.tuning_mode,
            "metric": args.metric,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "best_model_dir": str(best_model_dir),
            "copied_best_model_dir": str(final_dir),
        }
    )
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_params, file, ensure_ascii=False, indent=2)

    trials = [
        {"number": trial.number, "value": trial.value, "params": trial.params, "state": str(trial.state)}
        for trial in study.trials
    ]
    with (output_dir / "optuna_trials.json").open("w", encoding="utf-8") as file:
        json.dump(trials, file, ensure_ascii=False, indent=2)
    print(json.dumps(best_params, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

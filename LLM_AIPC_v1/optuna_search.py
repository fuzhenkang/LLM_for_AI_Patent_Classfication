"""Optuna hyperparameter search for LLM sequence classification."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from .llm_classifier import apply_model_defaults, train  # type: ignore  # noqa: E402
    from .llm_registry import MODEL_CONFIGS  # type: ignore  # noqa: E402
except ImportError:
    from llm_classifier import apply_model_defaults, train  # noqa: E402
    from llm_registry import MODEL_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for an LLM sequence classifier.")
    parser.add_argument("--model-key", default="qwen", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--text-cols", default=None)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--tuning-mode", default="qlora", choices=["lora", "qlora", "rslora", "dora", "head_only"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    base_args = apply_model_defaults(parse_args())
    output_dir = Path(base_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        args = argparse.Namespace(**vars(base_args))
        args.output_dir = str(output_dir / f"trial_{trial.number:03d}")
        args.max_len = trial.suggest_categorical("max_len", [128, 256, 384])
        args.batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
        args.lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        args.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        args.lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        args.lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        args.lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
        metrics = train(args)
        return float(metrics.get("f1_macro", 0.0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=base_args.n_trials)
    best_model_dir = output_dir / f"trial_{study.best_trial.number:03d}"
    best_params = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_model_dir": str(best_model_dir),
    }
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(best_params, file, ensure_ascii=False, indent=2)
    final_dir = output_dir / "best_model"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best_model_dir, final_dir)
    print(json.dumps(best_params, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

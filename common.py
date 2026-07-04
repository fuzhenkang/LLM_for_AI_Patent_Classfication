"""Shared utilities for standalone prompt-based patent classification."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


class SimpleLabelEncoder:
    def __init__(self) -> None:
        self.classes_: list[object] = []
        self._mapping: dict[str, int] = {}

    @staticmethod
    def _json_safe(label: object) -> object:
        if isinstance(label, np.generic):
            return label.item()
        return label

    @staticmethod
    def _key(label: object) -> str:
        return json.dumps(SimpleLabelEncoder._json_safe(label), ensure_ascii=False, sort_keys=True)

    def fit(self, labels: Sequence[object]) -> "SimpleLabelEncoder":
        self.classes_ = sorted({self._json_safe(label) for label in labels}, key=lambda item: str(item))
        self._mapping = {self._key(label): idx for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, labels: Sequence[object]) -> np.ndarray:
        return np.array([self._mapping[self._key(label)] for label in labels], dtype=np.int64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_label_encoder(*label_sequences: Sequence[object]) -> SimpleLabelEncoder:
    labels: list[object] = []
    for sequence in label_sequences:
        labels.extend(list(sequence))
    return SimpleLabelEncoder().fit(labels)


def save_label_encoder(encoder: SimpleLabelEncoder, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    classes = [SimpleLabelEncoder._json_safe(label) for label in encoder.classes_]
    with (output_path / "label_encoder.json").open("w", encoding="utf-8") as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)


def load_label_encoder(model_dir: str | Path) -> SimpleLabelEncoder:
    with (Path(model_dir) / "label_encoder.json").open("r", encoding="utf-8") as file:
        classes = json.load(file)
    encoder = SimpleLabelEncoder()
    encoder.classes_ = classes
    encoder._mapping = {SimpleLabelEncoder._key(label): idx for idx, label in enumerate(classes)}
    return encoder


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str]) -> dict[str, object]:
    true = np.array(y_true, dtype=np.int64)
    pred = np.array(y_pred, dtype=np.int64)
    accuracy = float((true == pred).mean()) if len(true) else 0.0

    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for idx, label in enumerate(label_names):
        tp = int(((true == idx) & (pred == idx)).sum())
        fp = int(((true != idx) & (pred == idx)).sum())
        fn = int(((true == idx) & (pred != idx)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((true == idx).sum()),
        }

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
    }


def write_metrics(metrics: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

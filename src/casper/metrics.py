"""Multi-label classification metrics for Stage 4."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np


def calculate_accuracy(
    true_classes: List[Dict],
    predicted_classes: List[Dict],
) -> Dict[str, float]:
    """Per-sample multi-label metrics: exact_match / precision / recall / f1 / jaccard."""
    true_ids = {c["class_id"] for c in true_classes}
    pred_ids = {c["class_id"] for c in predicted_classes} if predicted_classes else set()

    if not true_ids and not pred_ids:
        return {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}

    if not pred_ids and true_ids:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

    if pred_ids and not true_ids:
        # Predicted but no truth — recall is 1.0 by convention.
        return {"exact_match": 0.0, "precision": 0.0, "recall": 1.0, "f1": 0.0, "jaccard": 0.0}

    exact_match = 1.0 if true_ids == pred_ids else 0.0
    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    union = len(true_ids | pred_ids)
    jaccard = tp / union if union > 0 else 0.0

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


class MetricsAccumulator:
    """Streaming accumulator for sample-avg + micro + macro F1."""

    def __init__(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_exact_match = 0
        self.total_count = 0

        self.class_tp: Dict[int, int] = defaultdict(int)
        self.class_fp: Dict[int, int] = defaultdict(int)
        self.class_fn: Dict[int, int] = defaultdict(int)
        self.all_class_ids: set = set()

        self.sample_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

    def add_sample(self, true_classes: List[Dict], predicted_classes: List[Dict]) -> None:
        true_ids = {c["class_id"] for c in true_classes}
        pred_ids = {c["class_id"] for c in predicted_classes} if predicted_classes else set()

        self.all_class_ids.update(true_ids)
        self.all_class_ids.update(pred_ids)

        if true_ids == pred_ids:
            self.total_exact_match += 1

        tp = len(true_ids & pred_ids)
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        for class_id in (true_ids | pred_ids):
            if class_id in true_ids and class_id in pred_ids:
                self.class_tp[class_id] += 1
            elif class_id in pred_ids:
                self.class_fp[class_id] += 1
            else:
                self.class_fn[class_id] += 1

        self.total_count += 1

        sample_metric = calculate_accuracy(true_classes, predicted_classes)
        for key in self.sample_metrics:
            self.sample_metrics[key] += sample_metric[key]

    def get_micro_metrics(self) -> Dict[str, float]:
        denom_p = self.total_tp + self.total_fp
        denom_r = self.total_tp + self.total_fn
        precision = self.total_tp / denom_p if denom_p > 0 else 0.0
        recall = self.total_tp / denom_r if denom_r > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision_micro": precision, "recall_micro": recall, "f1_micro": f1}

    def get_macro_metrics(self) -> Dict[str, float]:
        if not self.all_class_ids:
            return {"precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

        precisions, recalls, f1s = [], [], []
        for class_id in self.all_class_ids:
            tp = self.class_tp[class_id]
            fp = self.class_fp[class_id]
            fn = self.class_fn[class_id]

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        return {
            "precision_macro": float(np.mean(precisions)),
            "recall_macro": float(np.mean(recalls)),
            "f1_macro": float(np.mean(f1s)),
        }

    def get_all_metrics(self) -> Dict[str, float]:
        if self.total_count == 0:
            return {
                "exact_match": 0.0,
                "precision_avg": 0.0,
                "recall_avg": 0.0,
                "f1_avg": 0.0,
                "jaccard_avg": 0.0,
                "precision_micro": 0.0,
                "recall_micro": 0.0,
                "f1_micro": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "total_samples": 0,
            }

        avg = {
            "exact_match": self.total_exact_match / self.total_count,
            "precision_avg": self.sample_metrics["precision"] / self.total_count,
            "recall_avg": self.sample_metrics["recall"] / self.total_count,
            "f1_avg": self.sample_metrics["f1"] / self.total_count,
            "jaccard_avg": self.sample_metrics["jaccard"] / self.total_count,
        }
        return {
            **avg,
            **self.get_micro_metrics(),
            **self.get_macro_metrics(),
            "total_samples": self.total_count,
        }

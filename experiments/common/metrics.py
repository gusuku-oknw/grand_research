"""Metrics computation utilities for SIS+pHash experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class RankingMetrics:
    precision_at: Dict[int, float]
    recall_at: Dict[int, float]
    map: float


def precision_recall_at_k(
    ranked_indices: Sequence[int],
    positive_indices: Sequence[int],
    k_values: Iterable[int],
) -> RankingMetrics:
    positives = set(positive_indices)
    total_positives = len(positives)
    prec: Dict[int, float] = {}
    rec: Dict[int, float] = {}
    hits = 0
    for k in sorted(k_values):
        for idx in ranked_indices[len(prec):k]:
            if idx in positives:
                hits += 1
        prec[k] = hits / k if k > 0 else 0.0
        rec[k] = hits / total_positives if total_positives > 0 else 0.0
    map_value = average_precision_from_ranking(ranked_indices, positives) if total_positives else 0.0
    return RankingMetrics(precision_at=prec, recall_at=rec, map=map_value)


def roc_pr_curves(distances: Sequence[int], labels: Sequence[int], tau_values: Iterable[int]) -> Tuple[List[float], List[float], List[float], List[float]]:
    labels_arr = np.asarray(labels, dtype=np.int32)
    pos = labels_arr == 1
    neg = ~pos
    tpr_list: List[float] = []
    fpr_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []
    total_pos = pos.sum()
    total_neg = neg.sum()
    for tau in tau_values:
        preds = np.asarray(distances) <= tau
        tp = np.logical_and(preds, pos).sum()
        fp = np.logical_and(preds, neg).sum()
        fn = total_pos - tp
        tn = total_neg - fp
        tpr = tp / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec_list.append(precision)
        rec_list.append(recall)
    return fpr_list, tpr_list, prec_list, rec_list


def average_precision_from_ranking(ranked_indices: Sequence[int], positives: set[int]) -> float:
    if not positives:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in positives:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(positives)


def area_under_curve(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.trapz(y, x))


def histogram(values: Sequence[int], bins: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins)
    return counts, edges

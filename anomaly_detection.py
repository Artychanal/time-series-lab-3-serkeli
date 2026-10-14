from __future__ import annotations

import numpy as np
import pandas as pd


def detect_and_compensate(values, window=12, threshold=4.0, compensate=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or window < 6 or threshold <= 0:
        raise ValueError("Invalid anomaly detector input or settings")
    cleaned = values.copy()
    scores = np.zeros(len(values))
    flags = np.zeros(len(values), dtype=bool)
    compensated = np.zeros(len(values), dtype=bool)
    medians = np.zeros(len(values))
    scales = np.zeros(len(values))
    for t in range(7, len(values)):


        history = np.diff(values[max(0, t - window - 1):t])
        median = float(np.median(history))
        sigma = 1.4826022185 * float(np.median(np.abs(history - median)))

        sigma = max(sigma, np.sqrt(np.finfo(float).eps) * max(abs(cleaned[t - 1]), 1.0))
        scores[t] = abs(values[t] - cleaned[t - 1] - median) / sigma
        medians[t], scales[t] = median, sigma
        flags[t] = scores[t] > threshold


        if compensate and flags[t] and not compensated[t - 1]:
            cleaned[t] = cleaned[t - 1] + median
            compensated[t] = True
    frame = pd.DataFrame({"observed": values, "processed": cleaned, "score": scores,
                          "flagged": flags, "compensated": compensated,
                          "past_median_change": medians, "past_robust_scale": scales})
    return cleaned, frame


def detector_metrics(truth, flags) -> dict:
    truth, flags = np.asarray(truth, bool), np.asarray(flags, bool)
    tp = int(np.sum(truth & flags))
    fp = int(np.sum(~truth & flags))
    fn = int(np.sum(truth & ~flags))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-15)}


def corruption_case(seed, n=69, center=9.0, sigma=0.08):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    clean = center + 0.004 * t + 0.05 * np.sin(t / 8) + rng.normal(0, sigma, n)
    truth = np.zeros(n, bool)

    positions = np.array([15, 29, 43, 58])
    positions = positions[positions < n]
    truth[positions] = True
    contaminated = clean.copy()
    contaminated[positions] += rng.choice([-1.0, 1.0], len(positions)) * rng.uniform(1.2, 2.0, len(positions))
    return clean, contaminated, truth


def select_detector(seed, n=69, center=9.0):
    rows = []
    for window in (6, 12, 18):
        for threshold in (3.0, 4.0, 5.0):
            truths, flags = [], []
            for replication in range(12):
                _, contaminated, truth = corruption_case(seed + replication, n, center)
                _, result = detect_and_compensate(contaminated, window, threshold, True)
                truths.extend(truth)
                flags.extend(result["flagged"])
            rows.append({"window": window, "threshold": threshold, **detector_metrics(truths, flags)})
    table = pd.DataFrame(rows).sort_values(["f1", "precision", "threshold", "window"], ascending=[False, False, False, False])
    best = table.iloc[0]
    return int(best["window"]), float(best["threshold"]), table


def evaluate_detector(seed, window, threshold, n=69, center=9.0):
    rows = []
    example = None
    for replication in range(30):
        clean, contaminated, truth = corruption_case(seed + replication, n, center)
        processed, detail = detect_and_compensate(contaminated, window, threshold, True)
        rows.append({"replication": replication, **detector_metrics(truth, detail["flagged"]),
                     "rmse_before": float(np.sqrt(np.mean((contaminated - clean) ** 2))),
                     "rmse_after": float(np.sqrt(np.mean((processed - clean) ** 2)))})
        if replication == 0:
            detail["clean_truth"] = clean
            detail["injected_error"] = truth
            example = detail
    return pd.DataFrame(rows), example

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import chi2

from kalman_filters import NoiseParameters, update


@dataclass(frozen=True)
class AdaptivePolicy:
    interval: int = 6
    adaptive_memory: bool = False
    windows: tuple[int, ...] = (18, 24, 36)
    validation_size: int = 6
    control: bool = False
    confidence: float = 0.99
    control_window: int = 3

    def __post_init__(self) -> None:
        if self.interval < 1 or self.validation_size < 2 or self.control_window < 2:
            raise ValueError("Invalid policy interval or validation/control window")
        if not 0 < self.confidence < 1 or any(w < self.validation_size + 6 for w in self.windows):
            raise ValueError("Invalid confidence or training memory candidates")


def choose_parameters(y, states, covariances, innovations, variances, qs,
                      current, process_base, transition, policy):
    n = len(y)
    validation = policy.validation_size
    windows = sorted(set([n] + [w for w in policy.windows if w <= n])) if policy.adaptive_memory else [n]
    candidates = []
    for window in windows:
        start, stop = max(1, n - window), n - validation
        if stop - start < 6:
            continue


        budgets = innovations[start:stop] ** 2 - variances[start:stop] + qs[start:stop] * process_base[0, 0]
        floor = np.finfo(float).eps * max(current.r, 1.0) / process_base[0, 0]
        q = max(float(np.mean(budgets) / process_base[0, 0]), floor)
        candidate = NoiseParameters(q, current.r)
        state, covariance = states[stop - 1].copy(), covariances[stop - 1].copy()
        errors = []
        for t in range(stop, n):
            state, covariance, prediction, *_ = update(state, covariance, y[t], transition, process_base, candidate)
            errors.append(prediction - y[t])
        candidates.append({"memory": int(window), "q": q, "r": current.r,
                           "validation_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                           "training_start": start, "training_stop_exclusive": stop,
                           "validation_stop_exclusive": n})
    if not candidates:
        return current, n, []
    best = min(candidates, key=lambda c: (c["validation_rmse"], -c["memory"]))
    return NoiseParameters(best["q"], current.r), best["memory"], candidates


@lru_cache(maxsize=32)
def control_limits(confidence, window):
    return chi2.ppf(0.95, df=1), chi2.ppf(confidence, df=window)


def control_inflation(nis_history, policy: AdaptivePolicy) -> float:
    if len(nis_history) < policy.control_window:
        return 1.0


    individual_limit, threshold = control_limits(policy.confidence, len(nis_history))
    if np.count_nonzero(nis_history > individual_limit) < 2:
        return 1.0
    return max(1.0, float(np.sum(nis_history) / threshold))

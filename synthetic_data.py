from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalman_filters import NoiseParameters, state_matrices

SCENARIOS = ("matched", "acceleration", "slope_change", "level_change", "measurement_noise", "outliers")


@dataclass
class SyntheticSeries:
    observations: np.ndarray
    truth: np.ndarray
    change_index: int | None
    measurement_std: np.ndarray
    injected: np.ndarray


def generate(scenario, seed, n=69, center=9.0, q=None, r=0.0144, order=2) -> SyntheticSeries:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    q = (1e-9 if order == 3 else 1e-5) if q is None else q
    NoiseParameters(q, r)
    transition, process_base, _ = state_matrices(order)
    rng = np.random.default_rng(seed)
    states = np.zeros((n, order))
    states[0, 0] = center
    change = n // 2
    for t in range(1, n):
        states[t] = transition @ states[t - 1] + rng.multivariate_normal(np.zeros(order), q * process_base)
        if scenario == "slope_change" and t == change:
            if order > 1:
                states[t, 1] += 0.05
            else:
                states[t, 0] += 0.5
        if scenario == "level_change" and t == change:
            states[t, 0] += 0.9
    if scenario == "acceleration":
        states[:, 0] += 0.0008 * np.arange(n) ** 2
    measurement_std = np.full(n, np.sqrt(r))
    if scenario == "measurement_noise":
        measurement_std[change:] *= 3
    observations = states[:, 0] + rng.normal(0, measurement_std)
    injected = np.zeros(n, bool)
    if scenario == "outliers":
        positions = np.array([15, 29, 43, 58])
        positions = positions[positions < n]
        observations[positions] += rng.choice([-1, 1], len(positions)) * 1.5
        injected[positions] = True
    return SyntheticSeries(observations, states[:, 0],
                           change if scenario in ("slope_change", "level_change", "measurement_noise") else None,
                           measurement_std, injected)

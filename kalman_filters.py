from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NoiseParameters:
    q: float
    r: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.q, self.r]).all() or self.q <= 0 or self.r <= 0:
            raise ValueError("q and r must be finite and strictly positive")


def state_matrices(order: int, dt: float = 1.0) -> tuple[NDArray, NDArray, NDArray]:
    if order not in (1, 2, 3) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("order must be 1, 2 or 3 and dt must be positive")
    transition = np.zeros((order, order))
    process_base = np.zeros_like(transition)
    for i in range(order):
        for j in range(order):
            if j >= i:
                transition[i, j] = dt ** (j - i) / factorial(j - i)
            a, b = order - 1 - i, order - 1 - j
            process_base[i, j] = dt ** (a + b + 1) / ((a + b + 1) * factorial(a) * factorial(b))
    measurement = np.zeros(order)
    measurement[0] = 1.0
    return transition, process_base, measurement


def update(state, covariance, observation, transition, process_base, parameters, inflation=1.0, fixed_gain=None):
    predicted_state = transition @ state
    predicted_covariance = transition @ covariance @ transition.T + parameters.q * process_base
    innovation = float(observation - predicted_state[0])

    innovation_variance = float(predicted_covariance[0, 0] + parameters.r)
    if not np.isfinite(innovation_variance) or innovation_variance <= 0:
        raise FloatingPointError("Invalid innovation covariance")
    corrected_prior = predicted_covariance * inflation
    gain = corrected_prior[:, 0] / (corrected_prior[0, 0] + parameters.r) if fixed_gain is None else fixed_gain
    corrected_state = predicted_state + gain * innovation
    error_matrix = np.eye(len(state))
    error_matrix[:, 0] -= gain
    corrected_covariance = error_matrix @ corrected_prior @ error_matrix.T + parameters.r * np.outer(gain, gain)
    corrected_covariance = (corrected_covariance + corrected_covariance.T) * 0.5
    if not np.isfinite(corrected_state).all() or not np.isfinite(corrected_covariance).all():
        raise FloatingPointError("Filter state or covariance became nonfinite")
    return corrected_state, corrected_covariance, float(predicted_state[0]), innovation, innovation_variance, gain


@dataclass
class FilterResult:
    state: NDArray
    covariance: NDArray
    prediction: NDArray
    innovation: NDArray
    innovation_variance: NDArray
    gain: NDArray
    q: NDArray
    r: NDArray
    memory: NDArray
    nis: NDArray
    inflation: NDArray
    control_triggered: NDArray
    parameter_history: list[dict]

    @property
    def estimate(self) -> NDArray:
        return self.state[:, 0]


def filter_series(observations, parameters, order, dt=1.0, initial_variance=None,
                  policy=None, adaptation_start=24, fixed_gain=None) -> FilterResult:
    from adaptive_filter import choose_parameters, control_inflation

    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or len(observations) < 2 or not np.isfinite(observations).all():
        raise ValueError("observations must be a finite 1D array with at least two values")
    transition, process_base, _ = state_matrices(order, dt)
    n = len(observations)
    states = np.zeros((n, order))
    covariances = np.zeros((n, order, order))
    predictions, innovations = np.zeros(n), np.zeros(n)
    variances = np.full(n, parameters.r)
    gains = np.zeros((n, order))
    qs, rs = np.full(n, parameters.q), np.full(n, parameters.r)
    memories = np.zeros(n, dtype=int)
    nis, inflation = np.zeros(n), np.ones(n)
    triggered = np.zeros(n, dtype=bool)

    states[0, 0] = observations[0]
    variance = parameters.r * 10 if initial_variance is None else float(initial_variance)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("initial_variance must be finite and positive")
    covariances[0] = np.eye(order) * variance
    covariances[0, 0, 0] = parameters.r
    predictions[0] = observations[0]
    current, chosen_memory = parameters, adaptation_start
    history: list[dict] = []
    for t in range(1, n):
        if policy is not None and t >= adaptation_start and (t - adaptation_start) % policy.interval == 0:
            current, chosen_memory, candidates = choose_parameters(
                observations[:t], states[:t], covariances[:t], innovations[:t],
                variances[:t], qs[:t], current, process_base, transition, policy)
            history.append({"index": t, "available_observations": t, "q": current.q,
                            "r": current.r, "memory": chosen_memory, "candidates": candidates})
        qs[t], rs[t], memories[t] = current.q, current.r, chosen_memory if policy is not None else t
        preliminary = update(states[t - 1], covariances[t - 1], observations[t], transition, process_base,
                             current, fixed_gain=fixed_gain)
        nis[t] = preliminary[3] ** 2 / preliminary[4]
        if policy is not None and policy.control and t >= adaptation_start:
            inflation[t] = control_inflation(nis[max(1, t - policy.control_window + 1):t + 1], policy)
            triggered[t] = inflation[t] > 1.0
        result = preliminary if inflation[t] == 1 else update(
            states[t - 1], covariances[t - 1], observations[t], transition, process_base, current, inflation[t])
        states[t], covariances[t], predictions[t], innovations[t], variances[t], gains[t] = result
    return FilterResult(states, covariances, predictions, innovations, variances, gains, qs, rs,
                        memories, nis, inflation, triggered, history)


def steady_state_gain(parameters, order, dt=1.0):
    from scipy.linalg import solve_discrete_are

    transition, process_base, measurement = state_matrices(order, dt)
    prior = solve_discrete_are(transition.T, measurement[:, None], parameters.q * process_base,
                              np.array([[parameters.r]]))
    return prior[:, 0] / (prior[0, 0] + parameters.r)


def gain_coefficients(gain, dt=1.0) -> dict[str, float]:
    result = {"alpha": float(gain[0])}
    if len(gain) >= 2:
        result["beta"] = float(gain[1] * dt)
    if len(gain) == 3:
        result["gamma"] = float(gain[2] * dt**2 / 2)
    return result


def forecast(result: FilterResult, horizon=3, dt=1.0) -> list[dict]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    state, covariance = result.state[-1].copy(), result.covariance[-1].copy()
    transition, process_base, _ = state_matrices(len(state), dt)
    rows = []
    for step in range(1, horizon + 1):
        state = transition @ state
        covariance = transition @ covariance @ transition.T + result.q[-1] * process_base
        variance = float(covariance[0, 0] + result.r[-1])
        sigma = np.sqrt(variance)
        rows.append({"horizon": step, "log_prediction": float(state[0]),
                     "log_variance": variance, "log_lower_95": float(state[0] - 1.95996398454 * sigma),
                     "log_upper_95": float(state[0] + 1.95996398454 * sigma),
                     "reviews_median": float(np.expm1(state[0])),
                     "reviews_mean_gaussian": float(np.expm1(state[0] + variance / 2)),
                     "reviews_lower_95": float(max(0, np.expm1(state[0] - 1.95996398454 * sigma))),
                     "reviews_upper_95": float(max(0, np.expm1(state[0] + 1.95996398454 * sigma)))})
    return rows

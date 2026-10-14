from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.optimize import minimize

from kalman_filters import NoiseParameters, state_matrices


def innovation_nll(log_parameters, observations, order, initial_variance) -> float:
    q, r = np.exp(log_parameters)
    transition, process_base, _ = state_matrices(order)
    state = np.zeros(order)
    state[0] = observations[0]
    covariance = np.eye(order) * initial_variance
    covariance[0, 0] = r
    total = 0.0
    identity = np.eye(order)
    for value in observations[1:]:
        state = transition @ state
        prior = transition @ covariance @ transition.T + q * process_base
        variance = prior[0, 0] + r
        innovation = value - state[0]
        total += np.log(2 * np.pi * variance) + innovation * innovation / variance
        gain = prior[:, 0] / variance
        state += gain * innovation
        error = identity.copy()
        error[:, 0] -= gain
        covariance = error @ prior @ error.T + r * np.outer(gain, gain)
    return float(total / 2)


def estimate_parameters(observations, order) -> tuple[NoiseParameters, dict]:
    observations = np.asarray(observations, dtype=float)
    if len(observations) < 12 or not np.isfinite(observations).all():
        raise ValueError("Parameter estimation requires at least 12 finite observations")
    scale = max(float(np.var(np.diff(observations))), float(np.var(observations)) * 1e-4, 1e-8)
    initial_variance = max(float(np.var(observations)), scale)
    bounds = [(np.log(scale * 1e-8), np.log(scale * 1e3))] * 2
    attempts = []
    for q_ratio, r_ratio in [(0.01, 0.5), (0.5, 0.1), (0.0001, 1.0)]:
        attempt = minimize(innovation_nll, np.log([scale * q_ratio, scale * r_ratio]),
                           args=(observations, order, initial_variance), method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 150, "ftol": 1e-9})
        if attempt.success and np.isfinite(attempt.fun):
            attempts.append(attempt)
    if not attempts:
        raise RuntimeError(f"Noise likelihood optimization failed for order {order}")
    best = min(attempts, key=lambda a: a.fun)
    params = NoiseParameters(*map(float, np.exp(best.x)))
    at_boundary = any(abs(best.x[i] - edge) < 0.01 for i in range(2) for edge in bounds[i])
    details = {**asdict(params), "order": order, "training_observations": len(observations),
               "negative_log_likelihood": float(best.fun), "optimizer_success": True,
               "successful_starts": len(attempts), "initial_variance": initial_variance,
               "boundary_solution": at_boundary,
               "interpretation": "Q/R separation may be weak; boundary solutions are explicitly reported"}
    return params, details

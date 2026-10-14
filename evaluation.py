from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from kalman_filters import filter_series


def metrics(actual, predicted) -> dict[str, float]:
    error = np.asarray(predicted, float) - np.asarray(actual, float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Metrics require nonempty finite paired observations")
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)), "error_std": float(np.std(error, ddof=1)) if len(error) > 1 else 0.0}


def forecast_metrics(actual_log, prediction_log):
    result = {f"{k}_log": v for k, v in metrics(actual_log, prediction_log).items()}

    original = metrics(np.expm1(actual_log), np.maximum(0, np.expm1(prediction_log)))
    result.update({f"{k}_reviews_median": v for k, v in original.items()})
    return result


def mean_hac(values):
    values = np.asarray(values, float)
    n = len(values)
    centered = values - np.mean(values)
    lag = max(1, min(int(n ** 0.25), n - 1))
    long_run_variance = float(centered @ centered / n)
    for k in range(1, lag + 1):
        long_run_variance += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k] / n)
    se = np.sqrt(max(long_run_variance, 0) / n)
    mean = float(np.mean(values))
    p = float(2 * stats.norm.sf(abs(mean / se))) if se > 0 else float(mean == 0)
    return {"mean": mean, "hac_standard_error": float(se), "hac_lag": lag,
            "mean_ci_lower": float(mean - 1.95996398454 * se),
            "mean_ci_upper": float(mean + 1.95996398454 * se), "mean_hac_pvalue": p}


def innovation_diagnostics(result, start=24):
    z = result.innovation[start:] / np.sqrt(result.innovation_variance[start:])
    if len(z) < 8:
        raise ValueError("At least eight innovations are needed for diagnostics")
    lag = min(6, max(1, len(z) // 5))
    lb = acorr_ljungbox(z, lags=[lag], return_df=True)
    acf1 = float(np.corrcoef(z[1:], z[:-1])[0, 1]) if np.std(z) > 0 else 0.0
    nis = result.nis[start:]
    return {"observations": len(z), **mean_hac(z), "std": float(np.std(z, ddof=1)),
            "acf1": acf1, "ljung_box_lag": lag,
            "ljung_box_statistic": float(lb["lb_stat"].iloc[0]),
            "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
            "nis_mean": float(np.mean(nis)),
            "nis_above_99_fraction": float(np.mean(nis > stats.chi2.ppf(0.99, 1))),
            "interpretation": "Approximate diagnostics; not a proof of unbiasedness of the hidden state"}


def recovery_steps(error, change, tolerance, consecutive=3):
    if change is None:
        return np.nan
    for t in range(change, len(error) - consecutive + 1):
        if np.all(np.abs(error[t:t + consecutive]) <= tolerance):
            return float(t - change)
    return np.nan


def monte_carlo(n, center, calibration, order, policies, simulations, seed, detector, progress=None,
                higher_order_parameters=None):
    from anomaly_detection import detect_and_compensate
    from synthetic_data import SCENARIOS, generate

    rows, examples = [], {}


    for scenario_index, scenario in enumerate(SCENARIOS):
        for replication in range(simulations):
            sample = generate(scenario, seed + scenario_index * 100_000 + replication, n, center, order=order)
            observations = sample.observations
            models = {"baseline": filter_series(observations, calibration, order)}
            if higher_order_parameters is not None:
                models["higher_order"] = filter_series(observations, higher_order_parameters, 3)
            for name, policy in policies.items():
                models[name] = filter_series(observations, calibration, order, policy=policy)

            if scenario == "outliers":
                processed, _ = detect_and_compensate(observations, *detector, compensate=True)
                models["compensated_adaptive"] = filter_series(processed, calibration, order, policy=policies["adaptive_full"])
            for name, result in models.items():
                error = result.estimate - sample.truth
                sl = slice(24, None)
                interval_half_width = 1.95996398454 * np.sqrt(result.covariance[:, 0, 0])
                row = {"scenario": scenario, "replication": replication, "model": name,
                       **metrics(sample.truth[sl], result.estimate[sl]),
                       "coverage_95": float(np.mean(np.abs(error[sl]) <= interval_half_width[sl])),
                       "recovery_steps": recovery_steps(error, sample.change_index, 3 * np.sqrt(0.0144)),
                       "control_count": int(result.control_triggered[sl].sum()),
                       "minimum_covariance_eigenvalue": float(np.min(np.linalg.eigvalsh(result.covariance))),
                       "max_absolute_state": float(np.max(np.abs(result.state)))}
                if sample.change_index is not None:
                    row["post_change_rmse"] = metrics(sample.truth[sample.change_index:], result.estimate[sample.change_index:])["rmse"]
                rows.append(row)
            if replication == 0:
                examples[scenario] = (sample, models)
        if progress is not None:
            progress(f"Monte Carlo: {scenario}, {simulations} реалізацій завершено")
    individual = pd.DataFrame(rows)
    summaries = []
    for (scenario, model), group in individual.groupby(["scenario", "model"], sort=False):
        biases = group["bias"].to_numpy()
        se = float(np.std(biases, ddof=1) / np.sqrt(len(group)))
        margin = float(stats.t.ppf(0.975, len(group) - 1) * se)
        mean_bias = float(np.mean(biases))
        recovery = group["recovery_steps"].dropna()
        summaries.append({"scenario": scenario, "model": model, "replications": len(group),
                          "mean_rmse": float(group["rmse"].mean()), "mean_mae": float(group["mae"].mean()),
                          "ensemble_bias": mean_bias, "bias_ci_lower": mean_bias - margin,
                          "bias_ci_upper": mean_bias + margin,
                          "bias_ci_contains_zero": bool(mean_bias - margin <= 0 <= mean_bias + margin),
                          "coverage_95": float(group["coverage_95"].mean()),
                          "mean_control_count": float(group["control_count"].mean()),
                          "recovered_fraction": len(recovery) / len(group) if scenario in ("slope_change", "level_change", "measurement_noise") else np.nan,
                          "recovery_median_if_recovered": float(recovery.median()) if len(recovery) else np.nan,
                          "minimum_covariance_eigenvalue": float(group["minimum_covariance_eigenvalue"].min())})
    return individual, pd.DataFrame(summaries), examples

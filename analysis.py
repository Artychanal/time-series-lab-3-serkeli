from __future__ import annotations

import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_filter import AdaptivePolicy
from anomaly_detection import detect_and_compensate, evaluate_detector, select_detector
from data_loader import load_data
from evaluation import forecast_metrics, innovation_diagnostics, metrics, monte_carlo
from kalman_filters import filter_series, forecast, gain_coefficients, steady_state_gain
from parameter_estimation import estimate_parameters
from synthetic_data import generate
from visualization import create_figures

INITIAL_TRAIN = 24
TEST_MONTHS = 12
MODEL_ORDER = {"alpha_beta": 2, "alpha_beta_gamma": 3, "level": 1,
               "fixed_alpha_beta": 2, "fixed_alpha_beta_gamma": 3}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, content):
    path.write_text(json.dumps(json_safe(content), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def model_run(y, parameters, details, order, name, adaptation_start, policy=None):
    fixed = steady_state_gain(parameters, order) if name.startswith("fixed_") else None
    return filter_series(y, parameters, order, initial_variance=details["initial_variance"],
                         policy=policy, adaptation_start=adaptation_start, fixed_gain=fixed)


def result_frame(result, dates, start=0):
    frame = pd.DataFrame({"date": dates, "prediction_log": result.prediction,
                          "estimate_log": result.estimate, "innovation": result.innovation,
                          "innovation_variance_pre_control": result.innovation_variance,
                          "nis_pre_control": result.nis, "posterior_variance": result.covariance[:, 0, 0],
                          "q": result.q, "r": result.r, "memory": result.memory,
                          "inflation": result.inflation, "control_triggered": result.control_triggered,
                          "alpha": result.gain[:, 0]})
    if result.state.shape[1] >= 2:
        frame["velocity"] = result.state[:, 1]
        frame["beta"] = result.gain[:, 1]
    if result.state.shape[1] == 3:
        frame["acceleration"] = result.state[:, 2]
        frame["gamma"] = result.gain[:, 2] / 2
    return frame.iloc[start:].reset_index(drop=True)


def run_analysis(data_path: Path, output: Path, simulations=300, seed=42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    log_messages = []

    def progress(message):
        print(message, flush=True)
        log_messages.append(message)

    progress("Завантаження та перевірка зафіксованого ряду Steam...")
    frame, quality = load_data(data_path)
    y = frame["log_reviews"].to_numpy(float)
    n, split = len(y), len(y) - TEST_MONTHS
    development = y[:split]
    progress(f"Повних місяців: {n}; розробка: {split}; фінальний тест: {TEST_MONTHS}.")
    window, threshold, detector_search = select_detector(seed + 1_000_000, n, float(np.median(development)))
    detector_test, cleaning_example = evaluate_detector(seed + 2_000_000, window, threshold, n, float(np.median(development)))
    _, anomalies = detect_and_compensate(y, window, threshold, compensate=False)
    anomalies.insert(0, "date", frame["date"])
    anomalies["decision"] = np.where(anomalies["flagged"], "retain: statistical flag is not proof of measurement error", "retain")
    detector_search.to_csv(output / "detector_parameter_search.csv", index=False)
    detector_test.to_csv(output / "detector_independent_validation.csv", index=False)
    cleaning_example.to_csv(output / "controlled_compensation_example.csv", index=False)
    anomalies.to_csv(output / "real_anomaly_candidates.csv", index=False)

    progress("Оцінювання Q/R та порівняння базових фільтрів на навчальній частині...")
    training_parameters, training_details = {}, {}
    for order in (1, 2, 3):
        training_parameters[order], training_details[order] = estimate_parameters(development[:INITIAL_TRAIN], order)
    cv_results, cv_rows = {}, []
    for name, order in MODEL_ORDER.items():
        result = model_run(development, training_parameters[order], training_details[order], order, name, INITIAL_TRAIN)
        cv_results[name] = result
        cv_rows.append({"model": name, **forecast_metrics(development[INITIAL_TRAIN:], result.prediction[INITIAL_TRAIN:])})
    naive_cv = development[INITIAL_TRAIN - 1:-1]
    cv_rows.append({"model": "naive", **forecast_metrics(development[INITIAL_TRAIN:], naive_cv)})
    order_name = min(("alpha_beta", "alpha_beta_gamma"), key=lambda name: next(r["rmse_log"] for r in cv_rows if r["model"] == name))
    selected_order = MODEL_ORDER[order_name]


    tuning_rows, candidate_results, candidates = [], {}, {}
    for interval in (3, 6, 12):
        for confidence in (0.95, 0.99):
            policy = AdaptivePolicy(interval=interval, adaptive_memory=True, control=True, confidence=confidence)
            result = model_run(development, training_parameters[selected_order], training_details[selected_order],
                               selected_order, "adaptive_full", INITIAL_TRAIN, policy)
            key = f"interval_{interval}_confidence_{confidence}"
            candidates[key], candidate_results[key] = policy, result
            tuning_rows.append({"candidate": key, **asdict(policy),
                                **forecast_metrics(development[INITIAL_TRAIN:], result.prediction[INITIAL_TRAIN:])})
    best_policy_key = min(tuning_rows, key=lambda row: (row["rmse_log"], -row["interval"], -row["confidence"]))["candidate"]
    full_policy = candidates[best_policy_key]
    policies = {
        "adaptive_parameters": replace(full_policy, adaptive_memory=False, control=False),
        "adaptive_memory": replace(full_policy, control=False),
        "adaptive_full": full_policy,
    }
    for name, policy in policies.items():
        result = candidate_results[best_policy_key] if name == "adaptive_full" else model_run(
            development, training_parameters[selected_order], training_details[selected_order],
            selected_order, name, INITIAL_TRAIN, policy)
        cv_results[name] = result
        cv_rows.append({"model": name, **forecast_metrics(development[INITIAL_TRAIN:], result.prediction[INITIAL_TRAIN:])})
    cv_table = pd.DataFrame(cv_rows)


    family = [order_name, "fixed_" + order_name, *policies]
    selected_name = min(family, key=lambda name: float(cv_table.loc[cv_table.model == name, "rmse_log"].iloc[0]))
    tuning_table = pd.DataFrame(tuning_rows)
    tuning_table.to_csv(output / "adaptive_policy_search.csv", index=False)
    cv_table.to_csv(output / "validation_comparison.csv", index=False)
    for name, result in cv_results.items():
        result_frame(result, frame["date"].iloc[:split], INITIAL_TRAIN).assign(model=name).to_csv(
            output / f"validation_{name}.csv", index=False)
    progress(f"До фінального тесту обрано: {selected_name}; базовий порядок: {selected_order}; політика: {best_policy_key}.")

    progress("Фінальний послідовний тест на останніх 12 місяцях...")
    test_parameters, test_details = {}, {}
    for order in (1, 2, 3):
        test_parameters[order], test_details[order] = estimate_parameters(development, order)
    test_results, test_rows, histories, diagnoses = {}, [], {}, {}
    all_models = {**MODEL_ORDER, **{name: selected_order for name in policies}}
    for name, order in all_models.items():
        result = model_run(y, test_parameters[order], test_details[order], order, name, split, policies.get(name))
        test_results[name] = result
        test_rows.append({"model": name, **forecast_metrics(y[split:], result.prediction[split:]),
                          "posterior_observation_rmse_log_descriptive_only": metrics(y[split:], result.estimate[split:])["rmse"],
                          "control_count": int(result.control_triggered[split:].sum())})
        diagnoses[name] = innovation_diagnostics(result, split)
        result_frame(result, frame["date"], split).assign(model=name).to_csv(output / f"test_{name}.csv", index=False)
        histories[name] = result.parameter_history
    test_rows.append({"model": "naive", **forecast_metrics(y[split:], y[split - 1:-1])})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(output / "test_model_comparison.csv", index=False)
    write_json(output / "adaptation_history.json", histories)
    write_json(output / "innovation_diagnostics.json", diagnoses)

    sensitivity_rows = []
    for order in (2, 3):

        reduced = development[1:]
        parameters, details = estimate_parameters(reduced[:INITIAL_TRAIN - 1], order)
        filtered = model_run(reduced, parameters, details, order, "sensitivity", INITIAL_TRAIN - 1)
        sensitivity_rows.append({"variant": "exclude_release_month", "order": order,
                                 **forecast_metrics(development[INITIAL_TRAIN:], filtered.prediction[INITIAL_TRAIN - 1:])})
        for factor in (0.1, 10.0):
            details = {**training_details[order], "initial_variance": training_details[order]["initial_variance"] * factor}
            filtered = model_run(development, training_parameters[order], details, order, "sensitivity", INITIAL_TRAIN)
            sensitivity_rows.append({"variant": f"initial_covariance_x{factor}", "order": order,
                                     **forecast_metrics(development[INITIAL_TRAIN:], filtered.prediction[INITIAL_TRAIN:])})
        raw = frame["total_reviews"].iloc[:split].to_numpy(float)
        parameters, details = estimate_parameters(raw[:INITIAL_TRAIN], order)
        filtered = model_run(raw, parameters, details, order, "raw_scale", INITIAL_TRAIN)
        raw_metrics = metrics(raw[INITIAL_TRAIN:], np.maximum(0, filtered.prediction[INITIAL_TRAIN:]))
        sensitivity_rows.append({"variant": "original_scale", "order": order,
                                 **{f"{k}_reviews_median": v for k, v in raw_metrics.items()},
                                 "scale_note": "Prediction on counts directly, not inverse log"})
    pd.DataFrame(sensitivity_rows).to_csv(output / "development_sensitivity.csv", index=False)

    progress("Незалежне синтетичне калібрування та перевірка відомого стану...")
    center = float(np.median(development))
    calibration_sample = generate("matched", seed + 3_000_000, n=180, center=center, order=selected_order)
    calibration, calibration_details = estimate_parameters(calibration_sample.observations, selected_order)
    higher_order_parameters, higher_order_details = (None, None)
    if selected_order == 2:
        higher_order_parameters, higher_order_details = estimate_parameters(calibration_sample.observations, 3)
    mc_individual, mc_summary, examples = monte_carlo(
        n, center, calibration, selected_order, policies, simulations, seed + 4_000_000,
        (window, threshold), progress, higher_order_parameters)
    mc_individual.to_csv(output / "monte_carlo_individual.csv", index=False)
    mc_summary.to_csv(output / "monte_carlo_summary.csv", index=False)
    for scenario, (sample, models) in examples.items():
        detail = pd.DataFrame({"index": np.arange(n), "truth": sample.truth,
                               "observation": sample.observations, "injected_error": sample.injected,
                               "measurement_std": sample.measurement_std})
        for name, result in models.items():
            detail[f"{name}_estimate"] = result.estimate
            detail[f"{name}_prediction"] = result.prediction
            detail[f"{name}_state_variance"] = result.covariance[:, 0, 0]
        detail.to_csv(output / f"synthetic_{scenario}.csv", index=False)
    if mc_individual["minimum_covariance_eigenvalue"].min() < -1e-8:
        raise FloatingPointError("Synthetic experiments found a non-positive covariance")


    final_parameters, final_details = estimate_parameters(y, selected_order)
    final_result = model_run(y, final_parameters, final_details, selected_order, selected_name,
                             INITIAL_TRAIN, policies.get(selected_name))
    if selected_name in policies:


        from adaptive_filter import choose_parameters
        from kalman_filters import state_matrices
        transition, process_base, _ = state_matrices(selected_order)
        next_params, _, _ = choose_parameters(y, final_result.state, final_result.covariance,
                                              final_result.innovation, final_result.innovation_variance,
                                              final_result.q, final_parameters, process_base, transition,
                                              policies[selected_name])
        final_result.q[-1] = next_params.q
    forecast_frame = pd.DataFrame(forecast(final_result))
    forecast_frame.insert(0, "date", pd.date_range(frame["date"].iloc[-1] + pd.offsets.MonthBegin(), periods=3, freq="MS"))
    forecast_frame.to_csv(output / "forecast_3_months.csv", index=False)
    progress("Побудова графіків та збереження підсумків...")
    create_figures(frame, anomalies, test_results, selected_name, cv_table, test_table,
                   mc_summary, examples, cleaning_example, forecast_frame, output / "figures")

    baseline_test = float(test_table.loc[test_table.model == order_name, "rmse_log"].iloc[0])
    adaptive_test = float(test_table.loc[test_table.model == "adaptive_full", "rmse_log"].iloc[0])
    summary = {
        "seed": seed, "simulations_per_scenario": simulations, "synthetic_scenarios": len(examples),
        "data_quality": quality,
        "protocol": {"initial_train": INITIAL_TRAIN, "development_observations": split,
                     "test_observations": TEST_MONTHS, "test_start": str(frame["date"].iloc[split].date()),
                     "test_end": str(frame["date"].iloc[-1].date()),
                     "evaluation": "One-step prediction before observation update, common dates for every model",
                     "test_policy": "Refit Q/R once on development; then ingest each test observation after scoring its prediction",
                     "warmup": "Initialization/calibration prefixes are not claimed as out-of-sample outputs"},
        "selected_order": selected_order, "selected_baseline": order_name,
        "selected_model_before_test": selected_name, "selected_policy": asdict(full_policy),
        "validation": cv_table.to_dict("records"), "test": test_table.to_dict("records"),
        "adaptive_full_test_rmse_change_percent": 100 * (adaptive_test / baseline_test - 1),
        "noise_estimation": {"initial_training": training_details, "development": test_details,
                             "final_known_history": final_details, "synthetic_independent_calibration": calibration_details},
        "higher_order_synthetic_calibration": higher_order_details,
        "final_gain_coefficients": gain_coefficients(final_result.gain[-1]),
        "anomalies": {"window": window, "threshold": threshold,
                      "real_flagged": int(anomalies["flagged"].sum()), "real_compensated": 0,
                      "selection": "12 independent synthetic training replications, maximum F1",
                      "evaluation": "30 disjoint synthetic validation replications",
                      "independent_mean_f1": float(detector_test.f1.mean()),
                      "independent_rmse_before": float(detector_test.rmse_before.mean()),
                      "independent_rmse_after": float(detector_test.rmse_after.mean())},
        "monte_carlo": mc_summary.to_dict("records"),
        "unbiasedness": {"theory": "Under correct linear dynamics, zero-mean noises, an unbiased initial state and deterministic gains, induction gives E[x_hat-x]=0. Data-dependent adaptive gains need empirical evaluation.",
                         "experiment": "Independent Gaussian state-space realizations; common independently estimated calibration; bias intervals computed across replications, not correlated time points",
                         "qualification": "An interval containing zero is not proof of exactly zero bias; the real latent state is unobserved; log-scale unbiasedness does not imply count-scale unbiasedness",
                         "multiple_comparisons": "95% intervals are pointwise, without family-wise multiplicity adjustment"},
        "limitations": ["Only 69 monthly observations; short final test",
                        "Q/R separation can be weak, especially near optimizer boundaries",
                        "Covariance matching can interpret increased measurement noise as process noise because R remains fixed",
                        "Chi-square control is approximate under non-Gaussian or correlated innovations",
                        "Synthetic calibration uncertainty is conditional on one independent training series",
                        "Forecast intervals are conditional Gaussian model intervals and exclude parameter uncertainty",
                        "The inverse-log point prediction used in comparisons is a median; Gaussian count mean is reported separately"],
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scipy", "matplotlib", "statsmodels")}},
    }
    write_json(output / "analysis_summary.json", summary)
    write_json(output / "data_provenance.json", {**quality, "source_project": "time_series_2", "local_file": data_path.name})
    progress("\nПідсумок фінального тесту:")
    progress(test_table[["model", "rmse_log", "mae_log", "bias_log"]].round(5).to_string(index=False))
    progress(f"Адаптивний фільтр: зміна RMSE проти {order_name}: {summary['adaptive_full_test_rmse_change_percent']:+.2f}% (від'ємне — покращення).")
    progress(f"Результати збережено: {output.resolve()}")
    (output / "run_log.txt").write_text("\n".join(log_messages) + "\n", encoding="utf-8")
    return summary

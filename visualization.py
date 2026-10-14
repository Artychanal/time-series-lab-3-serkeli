from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import chi2

LABELS = {"alpha_beta": "α–β Калман", "alpha_beta_gamma": "α–β–γ Калман",
          "fixed_alpha_beta": "α–β сталі коеф.", "fixed_alpha_beta_gamma": "α–β–γ сталі коеф.",
          "level": "Модель рівня", "naive": "Попереднє значення", "baseline": "Базовий фільтр",
          "adaptive_parameters": "Адаптація Q", "adaptive_memory": "Q та пам'ять",
          "adaptive_full": "Q, пам'ять і контроль", "compensated_adaptive": "Компенсація + адаптація",
          "higher_order": "Підвищення порядку: α–β–γ"}


def create_figures(frame, anomalies, test_results, selected_name, cv, comparison,
                   mc_summary, examples, cleaning, forecast_frame, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.22, "figure.dpi": 120, "savefig.dpi": 160})

    def save(name):
        plt.tight_layout()
        plt.savefig(output / name, bbox_inches="tight", facecolor="white")
        plt.close()

    dates, y = frame["date"], frame["total_reviews"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(dates, y, color="#1b8fc5")
    axes[0].set_ylabel("Відгуків / місяць")
    axes[0].set_title("Cyberpunk 2077: зафіксований місячний ряд")
    axes[1].plot(dates, frame["log_reviews"], label="log(1 + відгуки)")
    mask = anomalies["flagged"].to_numpy(bool)
    axes[1].scatter(dates[mask], frame.loc[mask, "log_reviews"], c="#cc5147", label="Статистичні кандидати; збережені")
    axes[1].set_ylabel("log(1 + відгуки)")
    axes[1].legend()
    save("01_real_series_anomalies.png")

    test_start = len(frame) - 12
    sl = slice(test_start, None)
    plt.figure(figsize=(11, 5))
    plt.plot(dates.iloc[sl], frame["log_reviews"].iloc[sl], "o-", color="black", label="Спостереження")
    for name in ("alpha_beta", "alpha_beta_gamma", "adaptive_full", "level"):
        plt.plot(dates.iloc[sl], test_results[name].prediction[sl], label=LABELS[name])
    plt.ylabel("log(1 + відгуки)")
    plt.title("Фінальний тест: однокрокові прогнози ДО корекції")
    plt.legend(ncol=2)
    save("02_test_predictions.png")

    plt.figure(figsize=(11, 5))
    plt.plot(dates.iloc[sl], frame["log_reviews"].iloc[sl], "o-", color="black", label="Вимірювання")
    for name in ("alpha_beta", "alpha_beta_gamma", "adaptive_full"):
        plt.plot(dates.iloc[sl], test_results[name].estimate[sl], label=LABELS[name])
    plt.ylabel("log(1 + відгуки)")
    plt.title("Фінальний тест: відфільтровані оцінки після корекції")
    plt.legend()
    save("03_filtered_test.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, table, title in [(axes[0], cv, "Валідація: вибір до фінального тесту"),
                             (axes[1], comparison, "Фінальний тест: 12 місяців")]:
        ordered = table.sort_values("rmse_log")
        ax.barh([LABELS.get(v, v) for v in ordered["model"]], ordered["rmse_log"], color="#1b8fc5")
        ax.invert_yaxis()
        ax.set_xlabel("RMSE однокрокового прогнозу, log")
        ax.set_title(title, fontsize=11)
    save("04_model_comparison.png")

    result = test_results[selected_name]
    innovations = result.innovation[sl] / np.sqrt(result.innovation_variance[sl])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(dates.iloc[sl], innovations, "o-")
    axes[0].axhline(0, color="black", ls="--")
    axes[0].set_title(f"Інновації фінального тесту: {LABELS[selected_name]}")
    axes[0].set_ylabel("Інновація / √S до контролю")
    plot_acf(innovations, lags=min(5, len(innovations) // 2 - 1), zero=False, ax=axes[1])
    axes[1].set_title("ACF інновацій; коротка тестова вибірка")
    axes[1].set_xlabel("Лаг, місяці")
    save("05_innovation_diagnostics.png")

    adaptive = test_results["adaptive_full"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(dates.iloc[sl], adaptive.nis[sl], "o-", label="NIS до втручання")
    axes[0].axhline(chi2.ppf(0.99, 1), ls="--", color="red", label="99% орієнтир для одного NIS")
    control = adaptive.control_triggered[sl]
    axes[0].scatter(dates.iloc[sl][control], adaptive.nis[sl][control], c="red", marker="x", s=70, label="Контроль невідповідності")
    axes[0].legend(fontsize=9)
    axes[0].set_ylabel("NIS")
    axes[1].plot(dates.iloc[sl], adaptive.q[sl], "o-")
    axes[1].set_ylabel("q")
    axes[2].step(dates.iloc[sl], adaptive.memory[sl], where="post")
    axes[2].set_ylabel("Пам'ять, місяці")
    axes[0].set_title("Адаптивний фільтр: контроль та історія параметрів")
    save("06_adaptive_history.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(cleaning["clean_truth"], color="black", label="Відомі правильні вимірювання")
    axes[0].plot(cleaning["observed"], alpha=0.65, label="Зі штучними помилками")
    axes[0].plot(cleaning["processed"], label="Причинна компенсація")
    axes[0].legend()
    axes[0].set_ylabel("log-шкала")
    axes[0].set_title("Компенсація помилок: незалежний контрольний сценарій")
    axes[1].plot(cleaning["score"])
    axes[1].set_ylabel("Робастне відхилення")
    axes[1].set_xlabel("Крок")
    save("07_controlled_anomaly_compensation.png")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, (scenario, (sample, models)) in zip(axes.flat, examples.items(), strict=True):
        ax.plot(sample.observations, color="gray", alpha=0.45, label="Спостереження")
        ax.plot(sample.truth, color="black", lw=2, label="Справжній стан")
        for name in ("baseline", "adaptive_full"):
            ax.plot(models[name].estimate, lw=1.3, label=LABELS[name])
        if "higher_order" in models:
            ax.plot(models["higher_order"].estimate, lw=1.0, ls="--", label=LABELS["higher_order"])
        if sample.change_index is not None:
            ax.axvline(sample.change_index, color="red", ls="--", alpha=0.6)
        ax.set_title(scenario)
        ax.set_xlabel("Крок")
    axes.flat[0].legend(fontsize=8)
    save("08_synthetic_scenarios.png")

    scenarios = mc_summary["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    comparison_models = ["baseline", "adaptive_parameters", "adaptive_memory", "adaptive_full"]
    if "higher_order" in mc_summary["model"].values:
        comparison_models.append("higher_order")
    width = 0.75 / len(comparison_models)
    for i, model in enumerate(comparison_models):
        subset = mc_summary[mc_summary["model"] == model].set_index("scenario").loc[scenarios]
        x = np.arange(len(scenarios)) + (i - (len(comparison_models) - 1) / 2) * width
        axes[0].bar(x, subset["mean_rmse"], width=width, label=LABELS[model])
        means = subset["ensemble_bias"].to_numpy()
        errors = np.vstack([means - subset["bias_ci_lower"], subset["bias_ci_upper"] - means])
        axes[1].errorbar(x, means, yerr=errors, fmt="o", capsize=2, label=LABELS[model])
    for ax in axes:
        ax.set_xticks(np.arange(len(scenarios)), scenarios, rotation=25, ha="right")
    axes[0].set_ylabel("RMSE стану, log")
    axes[0].set_title("Monte Carlo: середня помилка оцінки стану")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="black", ls="--")
    axes[1].set_title("Bias і 95% інтервал за незалежними реалізаціями")
    axes[1].set_ylabel("Зміщення, log")
    save("09_monte_carlo_rmse_bias.png")

    plt.figure(figsize=(11, 5))
    plt.plot(dates.iloc[-12:], y.iloc[-12:], "o-", label="Останні спостереження")
    plt.plot(forecast_frame["date"], forecast_frame["reviews_median"], "o--", label="Умовна медіана прогнозу")
    plt.fill_between(forecast_frame["date"], forecast_frame["reviews_lower_95"], forecast_frame["reviews_upper_95"], alpha=0.2, label="95% інтервал за Gaussian log-моделлю")
    plt.ylabel("Відгуків / місяць")
    plt.title("Додатковий прогноз на 3 місяці; умовні модельні інтервали")
    plt.legend(fontsize=9)
    save("10_forecast.png")

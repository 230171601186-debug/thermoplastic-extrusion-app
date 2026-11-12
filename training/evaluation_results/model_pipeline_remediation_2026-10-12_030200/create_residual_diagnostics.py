from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def build_band_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    stacked = predictions[
        predictions["split"].eq("outer_cv")
        & predictions["configuration"].eq("full_stack")
    ].copy()
    stacked["actual_band"] = np.select(
        [
            stacked["actual"] <= stacked["training_q25"],
            stacked["actual"] <= stacked["training_median"],
            stacked["actual"] <= stacked["training_q75"],
        ],
        ["lowest_quartile", "second_quartile", "third_quartile"],
        default="highest_quartile",
    )
    return (
        stacked.groupby(["target", "fold", "actual_band"], observed=True)
        .agg(
            n_rows=("actual", "size"),
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            residual_mean=("residual_pred_minus_actual", "mean"),
            residual_median=("residual_pred_minus_actual", "median"),
            MAE=("absolute_error", "mean"),
            residual_q10=("residual_pred_minus_actual", lambda x: x.quantile(0.1)),
            residual_q90=("residual_pred_minus_actual", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )


def build_process_residual_diagnostics() -> pd.DataFrame:
    predictions = pd.read_csv(BASE / "model1_outer_predictions.csv")
    return (
        predictions.groupby(["target", "fold"], sort=False)
        .agg(
            n_rows=("actual", "size"),
            target_min=("actual", "min"),
            target_max=("actual", "max"),
            target_std=("actual", "std"),
            residual_mean=("residual_pred_minus_actual", "mean"),
            residual_median=("residual_pred_minus_actual", "median"),
            residual_std=("residual_pred_minus_actual", "std"),
            residual_q05=("residual_pred_minus_actual", lambda x: x.quantile(0.05)),
            residual_q95=("residual_pred_minus_actual", lambda x: x.quantile(0.95)),
            absolute_error_median=("residual_pred_minus_actual", lambda x: x.abs().median()),
            absolute_error_q90=("residual_pred_minus_actual", lambda x: x.abs().quantile(0.90)),
        )
        .reset_index()
    )


def plot_target(predictions: pd.DataFrame, target: str) -> None:
    rows = predictions[
        predictions["split"].eq("outer_cv")
        & predictions["configuration"].eq("full_stack")
        & predictions["target"].eq(target)
    ]
    actual = rows["actual"].to_numpy()
    predicted = rows["predicted"].to_numpy()
    residual = rows["residual_pred_minus_actual"].to_numpy()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(actual, predicted, s=8, alpha=0.25)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Actual", ylabel="Predicted", title=f"{target}: actual vs predicted")
    axes[1].scatter(predicted, residual, s=8, alpha=0.25)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Predicted",
        ylabel="Prediction - actual",
        title=f"{target}: residual vs predicted",
    )
    figure.tight_layout()
    figure.savefig(BASE / f"{target.lower()}_outer_cv_residuals.png", dpi=140)
    plt.close(figure)


def main() -> None:
    predictions = pd.read_csv(BASE / "downstream_predictions_residuals.csv")
    build_band_diagnostics(predictions).to_csv(
        BASE / "downstream_residual_by_actual_band.csv", index=False
    )
    build_process_residual_diagnostics().to_csv(
        BASE / "model1_residual_diagnostics.csv", index=False
    )
    for target in ("Operational_Efficiency_pct", "Scrap_Rate"):
        plot_target(predictions, target)
    print("Created process/downstream residual diagnostics and two outer-CV plots.")


if __name__ == "__main__":
    main()

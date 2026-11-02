import matplotlib.pyplot as plt

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_predictions(resource, result):

    dates = result["test_dates"]

    actual = result["test_actual"]

    predicted = result["test_predictions"]

    anomalies = result["anomaly_flags"]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        dates,
        actual,
        marker="o",
        label="Actual Consumption"
    )

    ax.plot(
        dates,
        predicted,
        linestyle="--",
        marker="x",
        label="ML Predicted Consumption"
    )

    # Highlight detected anomalies.
    anomaly_dates = dates[anomalies]

    anomaly_values = actual[anomalies]

    if len(anomaly_dates) > 0:

        ax.scatter(
            anomaly_dates,
            anomaly_values,
            marker="x",
            s=120,
            label="Detected Anomaly"
        )

    ax.set_title(
        f"{resource.title()} - Actual vs Predicted Consumption"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Litres/day"
        if resource == "water"
        else "kWh/day"
    )

    ax.legend()

    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()

    fig.tight_layout()

    output_path = (
        REPORTS_DIR /
        f"{resource}_ml_prediction.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\nPrediction graph saved:", output_path)

    plt.show()

    plt.close(fig)


def plot_feature_importance(resource, model, features):

    if not hasattr(model, "feature_importances_"):

        print(
            "\nFeature importance visualization "
            "is available for Random Forest only."
        )

        return

    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        features,
        importance
    )

    ax.set_title(
        f"{resource.title()} - Feature Importance"
    )

    ax.set_xlabel("Importance")

    ax.set_ylabel("Features")

    fig.tight_layout()

    output_path = (
        REPORTS_DIR /
        f"{resource}_feature_importance.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\nFeature importance saved:", output_path)

    plt.show()

    plt.close(fig)
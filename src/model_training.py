import json
import joblib
import numpy as np

from pathlib import Path

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.feature_engineering import (
    create_features
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def train_model(df, resource):

    print("\n" + "=" * 60)

    print(f"TRAINING MODEL: {resource.upper()}")

    print("=" * 60)

    X, y, processed_df = create_features(
        df,
        resource
    )

    total_rows = len(X)

    train_end = int(total_rows * 0.70)

    validation_end = int(total_rows * 0.80)

    # Chronological split.
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[
        train_end:validation_end
    ]

    y_validation = y.iloc[
        train_end:validation_end
    ]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    print("\nDataset split:")

    print("Training records:", len(X_train))

    print("Validation records:", len(X_validation))

    print("Testing records:", len(X_test))

    # --------------------------------------------------
    # MODEL 1: LINEAR REGRESSION
    # --------------------------------------------------

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_predictions = linear_model.predict(
        X_validation
    )

    linear_mae = mean_absolute_error(
        y_validation,
        linear_predictions
    )

    # --------------------------------------------------
    # MODEL 2: RANDOM FOREST
    # --------------------------------------------------

    forest_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    forest_model.fit(
        X_train,
        y_train
    )

    forest_predictions = forest_model.predict(
        X_validation
    )

    forest_mae = mean_absolute_error(
        y_validation,
        forest_predictions
    )

    # --------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------

    print("\nVALIDATION PERFORMANCE")

    print(
        f"Linear Regression MAE: {linear_mae:.2f}"
    )

    print(
        f"Random Forest MAE: {forest_mae:.2f}"
    )

    if linear_mae <= forest_mae:

        best_model = linear_model

        model_name = "Linear Regression"

    else:

        best_model = forest_model

        model_name = "Random Forest"

    print("\nSelected Model:", model_name)

    # --------------------------------------------------
    # CALIBRATE ANOMALY THRESHOLD
    # --------------------------------------------------

    validation_predictions = best_model.predict(
        X_validation
    )

    validation_errors = np.abs(
        y_validation.to_numpy()
        - validation_predictions
    )

    # Use a conservative error threshold.
    error_threshold = max(
        float(np.quantile(validation_errors, 0.95)),
        float(y_train.mean() * 0.15)
    )

    # --------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------

    test_predictions = best_model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    print("\nFINAL TEST PERFORMANCE")

    for metric, value in metrics.items():

        print(f"{metric}: {value:.3f}")

    # --------------------------------------------------
    # TEST ANOMALY DETECTION
    # --------------------------------------------------

    test_errors = np.abs(
        y_test.to_numpy()
        - test_predictions
    )

    anomaly_flags = (
        test_errors > error_threshold
    )

    print(
        "\nAnomalies detected in test set:",
        int(anomaly_flags.sum())
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    model_path = (
        MODEL_DIR / f"{resource}_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    # Save model metadata.
    metadata = {
        "resource": resource,
        "model_name": model_name,
        "features": list(X.columns),
        "error_threshold": error_threshold,
        "validation_mae": float(
            min(linear_mae, forest_mae)
        ),
        "test_metrics": metrics
    }

    metadata_path = (
        MODEL_DIR / "model_metadata.json"
    )

    existing_metadata = {}

    if metadata_path.exists():

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            existing_metadata = json.load(file)

    existing_metadata[resource] = metadata

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            existing_metadata,
            file,
            indent=4
        )

    print("\nModel saved:", model_path)

    return {
        "model": best_model,
        "model_name": model_name,
        "metrics": metrics,
        "threshold": error_threshold,
        "test_actual": y_test,
        "test_predictions": test_predictions,
        "test_errors": test_errors,
        "anomaly_flags": anomaly_flags,
        "test_dates": processed_df.loc[
            X_test.index,
            "date"
        ]
    }
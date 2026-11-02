from src.data_loader import load_data
from src.model_training import train_model
from src.anomaly_detector import detect_anomaly
from src.investigator import investigate_cause
from src.visualizer import (
    plot_predictions,
    plot_feature_importance
)

def print_test_results(resource, result):

    print("\n" + "=" * 65)

    print(f"{resource.upper()} ML TEST RESULTS")

    print("=" * 65)

    print(
        f"Selected model: {result['model_name']}"
    )

    print("\nPerformance metrics:")

    for metric, value in result["metrics"].items():

        print(f"{metric}: {value:.3f}")

    print(
        "\nAnomalies detected:",
        int(result["anomaly_flags"].sum())
    )

    print("=" * 65)


def investigate_latest(resource, result, df):

    # The final observation in the chronological test set.
    actual = float(result["test_actual"].iloc[-1])

    predicted = float(
        result["test_predictions"][-1]
    )

    context_row = (
        df[df["resource"] == resource]
        .sort_values("date")
        .iloc[-1]
    )

    context = {

        "occupancy": int(context_row["occupancy"]),

        "temperature": float(context_row["temperature"]),

        "operating_hours": float(
            context_row["operating_hours"]
        )

    }

    # Detect the anomaly.
    anomaly_result = detect_anomaly(

        actual,

        predicted,

        result["threshold"]

    )

    # Investigate possible causes.
    investigation = investigate_cause(

        resource,

        anomaly_result,

        context

    )

    print("\n" + "=" * 65)

    print(f"{resource.upper()} LATEST INVESTIGATION")

    print("=" * 65)

    print("Actual consumption:", round(actual, 2))

    print("Predicted consumption:", round(predicted, 2))

    print(
        "Prediction error:",
        round(anomaly_result["residual"], 2)
    )

    print(
        "Anomaly threshold:",
        round(result["threshold"], 2)
    )

    print(
        "Anomaly detected:",
        anomaly_result["anomaly_detected"]
    )

    print(
        "Direction:",
        anomaly_result["direction"]
    )

    print("\nPossible explanations:")

    for cause in investigation["possible_causes"]:

        print("-", cause)

    print("\nRecommended actions:")

    for action in investigation["recommendations"]:

        print("-", action)

    print("\nConfidence and limitations:")

    print(
        investigation.get(
            "confidence",
            "No confirmed diagnosis available."
        )
    )

    print("=" * 65)


def main():

    print("\n")

    print("*" * 65)

    print("AI WATER AND ENERGY WASTE INVESTIGATOR")

    print("VERSION 5 - MACHINE LEARNING")

    print("*" * 65)

    print("\nLoading dataset...")

    df = load_data()

    print("Dataset loaded successfully.")

    print("Total records:", len(df))

    # Train independent models for each resource.
    results = {}

    for resource in ["water", "electricity"]:

        results[resource] = train_model(
            df,
            resource
        )

    while True:

        print("\nChoose an option:")

        print("1. View water ML results")

        print("2. View electricity ML results")

        print("3. Investigate water consumption")

        print("4. Investigate electricity consumption")

        print("5. View water prediction graph")

        print("6. View electricity prediction graph")

        print("7. View water feature importance")

        print("8. View electricity feature importance")

        print("9. Exit")

        choice = input(
            "\nEnter your choice: "
        ).strip()

        if choice == "1":

            print_test_results(
                "water",
                results["water"]
            )

        elif choice == "2":

            print_test_results(
                "electricity",
                results["electricity"]
            )

        elif choice == "3":

            investigate_latest(
                "water",
                results["water"],
                df
            )

        elif choice == "4":

            investigate_latest(
                "electricity",
                results["electricity"],
                df
            )

        elif choice == "5":

            plot_predictions(
                "water",
                results["water"]
            )

        elif choice == "6":

            plot_predictions(
                "electricity",
                results["electricity"]
            )

        elif choice == "7":

            result = results["water"]

            plot_feature_importance(
                "water",
                result["model"],
                result["model"].feature_names_in_
            )

        elif choice == "8":

            result = results["electricity"]

            plot_feature_importance(
                "electricity",
                result["model"],
                result["model"].feature_names_in_
            )

        elif choice == "9":

            print("\nApplication closed.")

            break

        else:

            print("Invalid choice.")


if __name__ == "__main__":

    main()
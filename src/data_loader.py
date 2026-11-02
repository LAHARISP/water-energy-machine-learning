import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "resource_consumption.csv"
)

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "Dataset not found. Run generate_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    required_columns = [
        "date",
        "resource",
        "consumption",
        "occupancy",
        "temperature",
        "operating_hours"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):

        raise ValueError(
            "Dataset is missing required columns."
        )

    if df[required_columns].isnull().any().any():

        raise ValueError(
            "Dataset contains missing values."
        )

    if (df["consumption"] < 0).any():

        raise ValueError(
            "Negative consumption detected."
        )

    if df.duplicated(
        subset=["date", "resource"]
    ).any():

        raise ValueError(
            "Duplicate date/resource records detected."
        )

    df = df.sort_values(
        ["resource", "date"]
    ).reset_index(drop=True)

    return df
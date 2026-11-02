import pandas as pd


FEATURE_COLUMNS = [
    "occupancy",
    "temperature",
    "operating_hours",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "rolling_mean_7"
]


def create_features(df, resource):

    resource_df = (
        df[df["resource"] == resource]
        .sort_values("date")
        .copy()
    )

    # Calendar features.
    resource_df["day_of_week"] = (
        resource_df["date"].dt.dayofweek
    )

    resource_df["month"] = (
        resource_df["date"].dt.month
    )

    resource_df["is_weekend"] = (
        resource_df["day_of_week"] >= 5
    ).astype(int)

    # Historical consumption features.
    resource_df["lag_1"] = (
        resource_df["consumption"].shift(1)
    )

    resource_df["rolling_mean_7"] = (
        resource_df["consumption"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # Remove rows without sufficient historical information.
    resource_df = resource_df.dropna().copy()

    # Water does not use equipment operating hours.
    if resource == "water":

        resource_df = resource_df.drop(
            columns=["operating_hours"]
        )

        features = [
            column
            for column in FEATURE_COLUMNS
            if column != "operating_hours"
        ]

    else:

        features = FEATURE_COLUMNS.copy()

    X = resource_df[features]

    y = resource_df["consumption"]

    return X, y, resource_df
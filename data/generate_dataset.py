import numpy as np
import pandas as pd
from pathlib import Path

# 1. CONFIGURATION
np.random.seed(42)
START_DATE = "2026-01-01"
DAYS = 180
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "resource_consumption.csv"
)

# 2. GENERATE DATES
dates = pd.date_range(
    start=START_DATE,
    periods=DAYS,
    freq="D"
)

# 3. GENERATE CONTEXTUAL FEATURES
occupancy = np.random.randint(
    65,
    101,
    DAYS
)

temperature = (
    27
    + 5 * np.sin(np.arange(DAYS) * 2 * np.pi / 365)
    + np.random.normal(0, 1.5, DAYS)
)

# Equipment operating hours.
operating_hours = np.random.uniform(
    7,
    13,
    DAYS
)

day_of_week = dates.dayofweek


# 4. GENERATE WATER CONSUMPTION

water_consumption = (
    450
    + occupancy * 9
    + temperature * 2
    + 12 * np.sin(2 * np.pi * day_of_week / 7)
    + np.random.normal(0, 20, DAYS)
)

water_consumption = np.maximum(
    water_consumption,
    100
)

# 5. GENERATE ELECTRICITY CONSUMPTION

electricity_consumption = (
    70
    + occupancy * 0.8
    + temperature * 2.5
    + operating_hours * 5
    + 8 * np.sin(2 * np.pi * day_of_week / 7)
    + np.random.normal(0, 8, DAYS)
)

electricity_consumption = np.maximum(
    electricity_consumption,
    20
)

# 6. INTRODUCE UNUSUAL CONSUMPTION EVENTS
# Inject anomalies into the final portion of the dataset.
# These are deliberately added for testing.
# Convert consumption values into mutable NumPy arrays

water_consumption = np.array(water_consumption, dtype=float)
electricity_consumption = np.array(electricity_consumption, dtype=float)

# Inject artificial anomalies into water consumption
water_consumption[-5] *= 1.65
water_consumption[-2] *= 1.90

# Inject artificial anomalies into electricity consumption
electricity_consumption[-7] *= 1.55
electricity_consumption[-3] *= 1.75


# 7. CREATE WATER DATAFRAME
water_df = pd.DataFrame({

    "date": dates,
    "resource": "water",
    "consumption": water_consumption,
    "occupancy": occupancy,
    "temperature": temperature,
    "operating_hours": 0

})

# 8. CREATE ELECTRICITY DATAFRAME

electricity_df = pd.DataFrame({

    "date": dates,
    "resource": "electricity",
    "consumption": electricity_consumption,
    "occupancy": occupancy,
    "temperature": temperature,
    "operating_hours": operating_hours

})

# 9. COMBINE BOTH DATASETS
df = pd.concat(
    [water_df, electricity_df],
    ignore_index=True
)

# 10. SAVE DATASET

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Dataset generated successfully!")
print("\nDataset shape:", df.shape)
print("\nResource distribution:")

print(
    df["resource"].value_counts()
)

print("\nSample records:")

print(df.head())

print("\nDataset saved to:", OUTPUT_PATH)
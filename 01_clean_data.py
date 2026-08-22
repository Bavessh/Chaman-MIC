"""
AI Travel Analyst — Part 1: Data Cleaning & Preprocessing
============================================================
Loads the raw flight pricing dataset and fixes every quality issue found
during inspection: padding blank rows, inconsistent text casing,
mixed units, currency symbols, word-numbers, and multiple duration/date
formats. Produces flight_data_cleaned.csv for the EDA step.
"""

import re
import numpy as np
import pandas as pd

RAW_PATH = "flight_pricing_dataset.csv"
OUT_PATH = "flight_data_cleaned.csv"

# ---------------------------------------------------------------
# 1. Load & drop the fully-blank padding rows at the bottom
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH, dtype=str)
before_rows = len(df)
df = df.dropna(how="all").reset_index(drop=True)
print(f"Dropped {before_rows - len(df)} fully-blank padding rows -> {len(df)} real rows remain")

# Drop exact duplicate flight records (same Flight_ID)
dupes = df.duplicated(subset="Flight_ID").sum()
df = df.drop_duplicates(subset="Flight_ID", keep="first").reset_index(drop=True)
print(f"Dropped {dupes} duplicate Flight_ID rows")

# ---------------------------------------------------------------
# 2. Airline — normalize casing to one canonical spelling
# ---------------------------------------------------------------
AIRLINE_MAP = {
    "indigo": "Indigo",
    "airasia india": "AirAsia India",
    "qatar airways": "Qatar Airways",
    "british airways": "British Airways",
    "gofirst": "GoFirst",
    "singapore airlines": "Singapore Airlines",
    "air india": "Air India",
    "emirates": "Emirates",
    "etihad airways": "Etihad Airways",
    "thai airways": "Thai Airways",
    "vistara": "Vistara",
    "lufthansa": "Lufthansa",
    "spicejet": "SpiceJet",
}
df["Airline"] = df["Airline"].str.strip().str.lower().map(AIRLINE_MAP)

# ---------------------------------------------------------------
# 3. Source / Destination — collapse IATA codes + "X Airport" + city
#    name into one canonical city name
# ---------------------------------------------------------------
IATA_MAP = {
    "AMD": "Ahmedabad", "BKK": "Bangkok", "BLR": "Bangalore", "BOM": "Mumbai",
    "CCU": "Kolkata", "DEL": "Delhi", "DOH": "Doha", "DXB": "Dubai",
    "FRA": "Frankfurt", "GOI": "Goa", "HYD": "Hyderabad", "JAI": "Jaipur",
    "JFK": "New York", "LHR": "London", "MAA": "Chennai", "PNQ": "Pune",
    "SIN": "Singapore", "SYD": "Sydney",
}

def canonical_city(val):
    if pd.isna(val):
        return np.nan
    v = val.strip()
    if v in IATA_MAP:
        return IATA_MAP[v]
    v = re.sub(r"\s+Airport$", "", v, flags=re.IGNORECASE).strip()
    return v

df["Source"] = df["Source"].apply(canonical_city)
df["Destination"] = df["Destination"].apply(canonical_city)

# ---------------------------------------------------------------
# 4. Total_Stops -> integer 0/1/2
# ---------------------------------------------------------------
STOPS_MAP = {"non-stop": 0, "1 stop": 1, "2 stops": 2}
def clean_stops(val):
    if pd.isna(val):
        return np.nan
    v = val.strip().lower()
    if v in STOPS_MAP:
        return STOPS_MAP[v]
    try:
        return int(v)
    except ValueError:
        return np.nan
df["Total_Stops"] = df["Total_Stops"].apply(clean_stops)

# ---------------------------------------------------------------
# 5. Passenger_Count -> integer (word numbers -> digits)
# ---------------------------------------------------------------
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
def clean_passengers(val):
    if pd.isna(val):
        return np.nan
    v = val.strip().lower()
    if v in WORD_NUM:
        return WORD_NUM[v]
    try:
        return int(v)
    except ValueError:
        return np.nan
df["Passenger_Count"] = df["Passenger_Count"].apply(clean_passengers)

# ---------------------------------------------------------------
# 6. Duration -> minutes (handles "Xh Ym", "X min", and decimal-hours)
# ---------------------------------------------------------------
def clean_duration(val):
    if pd.isna(val):
        return np.nan
    v = val.strip().lower()
    m = re.match(r"^(\d+)h\s*(\d+)m$", v)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.match(r"^(\d+)\s*min$", v)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+(\.\d+)?)$", v)
    if m:  # plain decimal number = hours
        return round(float(m.group(1)) * 60, 1)
    return np.nan
df["Duration_min"] = df["Duration"].apply(clean_duration)
df = df.drop(columns=["Duration"])

# ---------------------------------------------------------------
# 7. Distance_km -> float (strip " km" suffix)
# ---------------------------------------------------------------
df["Distance_km"] = (
    df["Distance_km"].astype(str).str.replace("km", "", regex=False).str.strip()
)
df["Distance_km"] = pd.to_numeric(df["Distance_km"], errors="coerce")

# ---------------------------------------------------------------
# 8. Price -> float (strip "Rs.", commas)
# ---------------------------------------------------------------
df["Price"] = (
    df["Price"].astype(str)
    .str.replace("Rs.", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# ---------------------------------------------------------------
# 9. Remaining numeric columns
# ---------------------------------------------------------------
df["Days_Before_Departure"] = pd.to_numeric(df["Days_Before_Departure"], errors="coerce")

# ---------------------------------------------------------------
# 10. Dates
# ---------------------------------------------------------------
df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], format="%m/%d/%Y", errors="coerce")

# ---------------------------------------------------------------
# 11. Handle missing values
#     - Price is the target variable: rows without it can't inform
#       price-driver analysis, so they're dropped.
#     - Remaining missing categoricals -> "Unknown" (keeps the row
#       usable for every OTHER analysis instead of discarding it).
#     - Remaining missing numerics -> median imputation.
# ---------------------------------------------------------------
before = len(df)
df = df.dropna(subset=["Price"]).reset_index(drop=True)
print(f"Dropped {before - len(df)} rows with missing Price (target variable)")

# Also drop nonsensical rows: price must be positive, duration/distance too
df = df[(df["Price"] > 0)].reset_index(drop=True)

categorical_cols = ["Airline", "Source", "Destination", "Travel_Class",
                     "Season", "Weekday", "Aircraft_Type", "Booking_Channel"]
for c in categorical_cols:
    df[c] = df[c].fillna("Unknown")

numeric_cols = ["Total_Stops", "Distance_km", "Days_Before_Departure",
                 "Passenger_Count", "Duration_min"]
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())

# ---------------------------------------------------------------
# 12. Outlier flag (informational, not removed) — cap prices likely
#     to be data-entry caps (exactly 200000 or 2000 repeated a lot)
# ---------------------------------------------------------------
print("\nRows at price=200000 (possible cap/outlier placeholder):", (df["Price"] == 200000).sum())
print("Rows at price=2000 (possible cap/outlier placeholder):", (df["Price"] == 2000).sum())

# ---------------------------------------------------------------
# 13. Derived feature used later in EDA/insights
# ---------------------------------------------------------------
df["Price_per_km"] = df["Price"] / df["Distance_km"].replace(0, np.nan)

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned dataset: {OUT_PATH}  shape={df.shape}")
print(df.dtypes)

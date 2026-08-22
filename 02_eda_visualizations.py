"""
AI Travel Analyst — Part 1: Exploration & Visualization
=========================================================
Generates 5+ visualizations and a factor analysis from the cleaned data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("flight_data_cleaned.csv")

# -----------------------------------------------------------
# Chart 1: Price distribution
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df["Price"], bins=50, ax=axes[0], color="#3b82f6")
axes[0].set_title("Distribution of Flight Prices")
axes[0].set_xlabel("Price (₹)")

sns.boxplot(x=df["Price"], ax=axes[1], color="#3b82f6")
axes[1].set_title("Flight Price — Boxplot (outlier view)")
axes[1].set_xlabel("Price (₹)")
plt.tight_layout()
plt.savefig("chart1_price_distribution.png")
plt.close()

# -----------------------------------------------------------
# Chart 2: Price by Airline
# -----------------------------------------------------------
order = df.groupby("Airline")["Price"].median().sort_values(ascending=False).index
plt.figure(figsize=(11, 6))
sns.boxplot(data=df, x="Airline", y="Price", order=order, palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.title("Flight Price by Airline")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("chart2_price_by_airline.png")
plt.close()

# -----------------------------------------------------------
# Chart 3: Price vs Days Before Departure
# -----------------------------------------------------------
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Days_Before_Departure", y="Price", alpha=0.35, s=20, color="#ef4444")
sns.regplot(data=df, x="Days_Before_Departure", y="Price", scatter=False, color="black", order=2)
plt.title("Price vs. Days Before Departure")
plt.xlabel("Days Before Departure (booking lead time)")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("chart3_price_vs_lead_time.png")
plt.close()

# -----------------------------------------------------------
# Chart 4: Price by Number of Stops
# -----------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Total_Stops", y="Price", palette="crest")
plt.title("Flight Price by Number of Stops")
plt.xlabel("Total Stops")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("chart4_price_by_stops.png")
plt.close()

# -----------------------------------------------------------
# Chart 5: Price by Travel Class
# -----------------------------------------------------------
class_order = ["Economy", "Premium Economy", "Business", "First"]
class_order = [c for c in class_order if c in df["Travel_Class"].unique()]
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x="Travel_Class", y="Price", order=class_order, palette="magma")
plt.title("Flight Price by Travel Class")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("chart5_price_by_class.png")
plt.close()

# -----------------------------------------------------------
# Chart 6: Price vs Distance (with class coloring)
# -----------------------------------------------------------
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Distance_km", y="Price", hue="Travel_Class",
                 hue_order=class_order, alpha=0.5, s=25, palette="tab10")
plt.title("Price vs. Distance, by Travel Class")
plt.xlabel("Distance (km)")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig("chart6_price_vs_distance.png")
plt.close()

# -----------------------------------------------------------
# Chart 7: Average price by season & weekday (heatmap)
# -----------------------------------------------------------
pivot = df.pivot_table(index="Season", columns="Weekday", values="Price", aggfunc="median")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = pivot.reindex(columns=[d for d in weekday_order if d in pivot.columns])
plt.figure(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Median Price (₹)"})
plt.title("Median Price by Season and Weekday")
plt.tight_layout()
plt.savefig("chart7_season_weekday_heatmap.png")
plt.close()

# -----------------------------------------------------------
# Chart 8: Booking Channel comparison
# -----------------------------------------------------------
plt.figure(figsize=(9, 6))
order2 = df.groupby("Booking_Channel")["Price"].median().sort_values(ascending=False).index
sns.barplot(data=df, x="Booking_Channel", y="Price", order=order2, palette="flare", estimator=np.median)
plt.xticks(rotation=30, ha="right")
plt.title("Median Price by Booking Channel")
plt.ylabel("Median Price (₹)")
plt.tight_layout()
plt.savefig("chart8_booking_channel.png")
plt.close()

print("Saved 8 charts.")

# -----------------------------------------------------------
# Correlation matrix for numeric factors
# -----------------------------------------------------------
num_cols = ["Price", "Distance_km", "Duration_min", "Total_Stops",
            "Days_Before_Departure", "Passenger_Count"]
corr = df[num_cols].corr(numeric_only=True)
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation Matrix — Numeric Factors vs Price")
plt.tight_layout()
plt.savefig("chart9_correlation_matrix.png")
plt.close()
print("Correlation with Price:")
print(corr["Price"].sort_values(ascending=False))

# -----------------------------------------------------------
# Categorical factor strength: eta-squared-ish via group means spread
# -----------------------------------------------------------
print("\n--- Median price by category (spread = max/min ratio) ---")
for col in ["Airline", "Travel_Class", "Total_Stops", "Season", "Booking_Channel"]:
    med = df.groupby(col)["Price"].median().sort_values(ascending=False)
    ratio = med.max() / med.min()
    print(f"{col}: max/min median price ratio = {ratio:.2f}x")
    print(med, "\n")

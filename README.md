# AI Travel Analyst — Part 1: Exploration

## Project Overview
This project is a data-driven analysis of flight pricing data, built as Part 1 (Exploration)
of the AI Travel Analyst assignment. It cleans a messy real-world-style flight dataset,
visualizes pricing patterns, and identifies which factors most strongly influence flight prices.

## Problem Statement
Flight prices are influenced by many interacting factors — route, airline, travel class,
timing, and more — but it's not obvious from raw booking data which of these actually move
the price and which don't. The goal of this project is to clean a raw flight pricing dataset,
explore it visually, and answer: **what actually drives flight prices, and what does that
mean for travelers?**

## Installation Instructions
1. Make sure Python 3.10+ is installed.
2. Install the required libraries:
   ```
   pip install pandas numpy matplotlib seaborn
   ```
3. Place `flight_pricing_dataset.csv` (the raw dataset) in the same folder as the scripts.
4. Run the cleaning script:
   ```
   python 01_clean_data.py
   ```
   This produces `flight_data_cleaned.csv`.
5. Run the visualization script:
   ```
   python 02_eda_visualizations.py
   ```
   This produces 9 chart PNGs and prints the correlation/factor analysis to the console.
6. Open `AI_Travel_Analyst_Part1_Report.html` in any browser to view the full written report
   with all charts and insights embedded.

## Dataset Used
`flight_pricing_dataset.csv` — a flight booking dataset with 18 fields per record, including
Airline, Source, Destination, Departure Date/Time, Arrival Time, Duration, Total Stops,
Distance (km), Travel Class, Days Before Departure, Season, Weekday, Aircraft Type, Booking
Channel, Passenger Count, and Price.

The raw file contained 100,000 rows, of which 95,001 were fully blank padding rows. After
removing those, 4,999 real records remained; after further cleaning (removing 1 duplicate and
268 rows with a missing price), the final cleaned dataset has **4,730 records**.

## Methodology
1. **Data Cleaning** (`01_clean_data.py`)
   - Removed blank padding rows and duplicate `Flight_ID` entries
   - Standardized inconsistent airline name casing (e.g. "AIR INDIA", "air india", "Air India" → one spelling)
   - Collapsed mixed location formats (IATA codes, "City Airport", plain city names) into one canonical city name per Source/Destination
   - Standardized `Total_Stops` (mixed "non-stop"/"1 stop"/"2 stops"/raw digits → integers)
   - Converted word-based passenger counts ("two", "five") to integers
   - Converted three different `Duration` formats (decimal hours, "Xh Ym", "X min") into a single minutes value
   - Stripped unit/currency text from `Distance_km` ("298.6 km") and `Price` ("Rs. 45,690.74")
   - Handled missing values: dropped rows missing the target (`Price`); filled missing categoricals with "Unknown"; median-imputed missing numerics
   - Flagged a data-quality issue: ~490 rows priced at exactly ₹200,000 and ~100 at exactly ₹2,000, suggesting artificial caps rather than real fares

2. **Exploration & Visualization** (`02_eda_visualizations.py`)
   - Price distribution (histogram + boxplot)
   - Price by airline, travel class, number of stops, booking channel
   - Price vs. booking lead time and vs. distance
   - Season/weekday price heatmap
   - Correlation matrix of numeric factors against price
   - Factor-strength comparison (spread ratios across categories)

## Technologies Used
- Python 3
- pandas, NumPy — data cleaning and analysis
- matplotlib, seaborn — visualization
- HTML/CSS — final report formatting

## Results
- **Distance** and **Duration** are the strongest numeric drivers of price (correlation 0.69 and 0.67 with Price).
- **Travel Class** shows the clearest categorical separation — median First Class fare is ~4.8x Economy.
- **Airline** shows a large price spread (16.7x), but this largely reflects route type (international full-service vs. budget domestic) rather than the carrier itself.
- **Number of stops** has a moderate effect (2.0x spread), partly confounded with route distance.
- **Booking lead time**, **season**, and **booking channel** have only weak effects on price.
- **Passenger count** has essentially no effect on per-ticket price (correlation 0.005).

Full charts and detailed discussion are in `AI_Travel_Analyst_Part1_Report.html`.

## Challenges Faced
- The raw file was 95% blank padding rows, which had to be identified and removed before any real analysis could begin.
- Airline names, city names, stop counts, passenger counts, and durations were each recorded in 2–4 inconsistent formats, requiring custom parsing logic for each column.
- Price values were inconsistently formatted, with some including currency symbols and thousands separators as text.
- A cluster of suspiciously round price values (exactly ₹200,000 and ₹2,000) suggested synthetic data caps, which had to be identified and flagged rather than silently treated as normal prices.

## Future Improvements
- Investigate and properly handle the ₹200,000 / ₹2,000 price-cap cluster before using this data for modeling.
- Extend to Part 2: feature engineering and a price-prediction model using distance, duration, and travel class as core features.
- Add interactive filtering (e.g. an interactive dashboard) to let users explore price by route pair directly.

## Demo
Demo link: 
https://youtu.be/KispZaREP0M

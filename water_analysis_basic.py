# Project 25: Water Usage & Conservation Analysis (basic version)
import pandas as pd
import numpy as np

# Task 1: load the data and look at it
df = pd.read_csv("water_consumption_expanded.csv")
print("Task 1")
print(df.shape)
print(df.columns)
print(df.dtypes)

# Task 2: missing values and duplicate meter IDs
print("\nTask 2")
print(df.isnull().sum())
print("Duplicate Meter_Record_ID:", df["Meter_Record_ID"].duplicated().sum())
df = df.drop_duplicates(subset="Meter_Record_ID")

# Task 3: fill missing usage with the median of each household type
# (median is not affected by very big or very small values)
print("\nTask 3")
medians = df.groupby("Household_Type")["Water_Usage_Liters"].transform("median")
df["Water_Usage_Liters"] = df["Water_Usage_Liters"].fillna(medians)
print("Missing now:", df["Water_Usage_Liters"].isnull().sum())

# Task 4: dates, month and weekday
print("\nTask 4")
df["Reading_Date"] = pd.to_datetime(df["Reading_Date"])
df["Month"] = df["Reading_Date"].dt.month_name()
df["Year_Month"] = df["Reading_Date"].dt.to_period("M")
df["Weekday"] = df["Reading_Date"].dt.day_name()
print(df[["Reading_Date", "Month", "Weekday"]].head())

# Task 5: total and average per household
print("\nTask 5")
print("Total water used:", df["Water_Usage_Liters"].sum())
household = df.groupby("Household_ID")["Water_Usage_Liters"].agg(["sum", "mean"])
print(household.head())

# Task 6: month to month change for each household
print("\nTask 6")
monthly = df.groupby(["Household_ID", "Year_Month"])["Water_Usage_Liters"].sum().reset_index()
monthly["Change"] = monthly.groupby("Household_ID")["Water_Usage_Liters"].diff()
print(monthly.dropna().head(10))

# Task 7: rank household types by total usage
print("\nTask 7")
type_total = df.groupby("Household_Type")["Water_Usage_Liters"].sum().sort_values(ascending=False)
print(type_total)

# Task 8: average usage by building type and zone
print("\nTask 8")
print(df.groupby("Building_Type")["Water_Usage_Liters"].mean())
print(df.groupby("Zone")["Water_Usage_Liters"].mean())

# Task 9: top 10 households
print("\nTask 9")
print(household.sort_values("sum", ascending=False).head(10))

# Task 10: month with highest usage
print("\nTask 10")
month_total = df.groupby("Year_Month")["Water_Usage_Liters"].sum()
print("Highest month:", month_total.idxmax(), month_total.max())

# Task 11: dictionary of zone totals and set of unique zones
print("\nTask 11")
zone_totals = {}
for i in range(len(df)):
    zone = df.iloc[i]["Zone"]
    usage = df.iloc[i]["Water_Usage_Liters"]
    if zone in zone_totals:
        zone_totals[zone] += usage
    else:
        zone_totals[zone] = usage
print(zone_totals)
print(set(df["Zone"]))

# Task 12: function to classify usage
# Low = below 25th percentile, High = above 75th percentile, rest Normal
print("\nTask 12")
low_limit = np.percentile(df["Water_Usage_Liters"], 25)
high_limit = np.percentile(df["Water_Usage_Liters"], 75)

def classify(usage):
    if usage < low_limit:
        return "Low"
    elif usage > high_limit:
        return "High"
    else:
        return "Normal"

df["Category"] = df["Water_Usage_Liters"].apply(classify)
print(low_limit, high_limit)
print(df["Category"].value_counts())

# Task 13: statistics with numpy
print("\nTask 13")
values = df["Water_Usage_Liters"].values
print("Mean:", np.mean(values))
print("Median:", np.median(values))
print("Variance:", np.var(values))
print("Std dev:", np.std(values))

# Task 14: outliers using IQR
print("\nTask 14")
q1 = np.percentile(values, 25)
q3 = np.percentile(values, 75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
outliers = df[df["Water_Usage_Liters"] > upper]
print("Upper limit:", upper)
print("Number of outliers:", len(outliers))
print(df.groupby("Household_Type")["Water_Usage_Liters"].mean())

# Task 15: WaterMeter class and recommendations
print("\nTask 15")
class WaterMeter:
    def __init__(self, meter_id, household_type):
        self.meter_id = meter_id
        self.household_type = household_type

    def show(self):
        print("Meter:", self.meter_id, "| Type:", self.household_type)

meter = WaterMeter(df.iloc[0]["Meter_Record_ID"], df.iloc[0]["Household_Type"])
meter.show()

print("""
Recommendations (update these using your own results):
1. Focus awareness campaigns on the household type with the highest total usage.
2. Check for leaks in the zone with the highest average usage.
3. Send alerts to households in the 'High' category.
4. Run saving campaigns before the month with the highest usage.
5. Install low-flow taps and smart meters in the building type that uses the most water.
""")

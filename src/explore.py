import pandas as pd

df = pd.read_csv("data/retail_price.csv")

# --- Basic shape and structure ---

# dataframe shape
print("Shape of DF:", df.shape)

# list of df columns
print("List of columns:", list(df.columns))

# top 5 rows of df
print("Top 5 Rows:\n", df.head())

# column types
print("Data types of columns:\n", df.dtypes)

# --- Selecting columns ---

# a single column comes back as a Series
prices = df["unit_price"]
print(type(prices))
print("Top 5 rows unit_price values:\n", prices.head())

# a list of columns comes back as a DataFrame
print(df[["product_id", "month_year", "qty", "customers"]].head(30))

# --- Indexing ---

print(df.index)
print(df.loc[0])

# setting index to a specific column
df_indexed = df.set_index("product_id")
print(df_indexed.head())

# --- Filtering using conditions ---

df_qnt = df[df["qty"] > 5]
print("Shape of df where qty sold > 5:", df_qnt.shape)

# combining conditions needs & (not "and") and parentheses around each condition
df_filter_condition = df[(df["product_category_name"] == "bed_bath_table") & (df["unit_price"] > 40)]
print("Combined condition filter shape:", df_filter_condition.shape)

# full price history for one product — this is filtering, not grouping
df_bed1 = df[df["product_id"] == "bed1"]
print("bed1 info:\n", df_bed1[["month_year", "unit_price", "comp_1", "comp_2", "comp_3"]])

# --- Data quality checks ---

print("Null values per column:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print("Dataframe summary statistics:\n", df.describe())

# --- Grouping ---

# average unit price per category
print("Average unit_price per category:\n", df.groupby("product_category_name")["unit_price"].mean())

# average of several numeric columns per product.
# column selection needs a LIST, and a groupby needs a terminal operation
# like .mean() — without one you get a DataFrameGroupBy object, not data.
# month_year is excluded because a date string can't be averaged.
product_summary = df.groupby("product_id")[
    ["unit_price", "freight_price", "product_score", "comp_2", "fp2", "ps2"]
].mean()
print("Per-product averages:\n", product_summary.head())
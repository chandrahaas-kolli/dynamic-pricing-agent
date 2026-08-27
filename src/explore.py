import pandas as pd

df = pd.read_csv("data/retail_price.csv")

#dataframe shape
print("Shape of DF:", df.shape)

#list of df columns
print("List of columns:", list(df.columns))

#top 5 rows of df
print("Top 5 Rows:", df.head())

#accessing unit price of products
prices = df["unit_price"]

#type of columns
print(type(prices))
print("Top 5 rows unit_price values", prices.head())

#indexing
print(df.index)
print(df.loc[0])

#setting index to a specific column
df_indexed = df.set_index("product_id")
print(df_indexed.head())

#filtering using conditions
df_qnt = df[df["qty"] > 5]
print("Shape of df where qty sold > 5:", df_qnt.shape)

df_filter_condition = df[(df["product_category_name"] == "bed_bath_table") & (df["unit_price"] > 40)]
print("Combined condition filter shape:", df_filter_condition.shape)

df_bed3 = df[df["product_id"] == "bed3"]
print("bed3 info:\n", df_bed3[["month_year", "unit_price", "comp_1", "comp_2", "comp_3"]])

#checking for null values
print("Null values per column:\n", df.isnull().sum())

#checking for duplicates
print("Duplicates:", df.duplicated().sum())

#summary statistics
print("Dataframe summary statistics:\n", df.describe())

#column types
print("Data types of columns:\n", df.dtypes)

#grouping
print("Unit_price category avg price\n", df.groupby("product_category_name")["unit_price"].mean())
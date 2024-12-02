import sqlite3
import pandas as pd

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('sales_analysis.csv')

# Check for missing values in the dataset
print("Missing values before handling:")
print(df.isnull().sum())

# Handle missing values:
# - For numeric columns, fill NaN with the mean
# - For categorical columns, fill NaN with the mode
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)

# Verify that there are no missing values
print("Missing values after handling:")
print(df.isnull().sum())

# Connect to SQLite database (it will create a new database if not existing)
# Reconnect to the database
conn = sqlite3.connect('sales_data.db')

# Query the database to verify the data
query = "SELECT * FROM sales LIMIT 5;"  # Adjust LIMIT as needed
result = pd.read_sql_query(query, conn)

# Display the queried data
print("Sample data from the 'sales' table:")
print(result)

# Close the connection
conn.close()


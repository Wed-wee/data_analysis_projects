import pandas as pd
import sqlite3
print("SQLite version:", sqlite3.sqlite_version)
# Connect to SQLite database (replace with your actual database file)
conn = sqlite3.connect('sales_data.db')  # Path to your SQLite database file

# Query the data from the database
query = "SELECT * FROM sales"  # Replace 'sales' with your actual table name
data = pd.read_sql(query, conn)

# Close the connection to the database
conn.close()

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Handle missing data by filling missing sales values with the mean
data['Sales'] = data['Sales'].fillna(data['Sales'].mean())

# Feature engineering
data['Day_of_Week'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Week_of_Year'] = data['Date'].dt.isocalendar().week

# Lag feature: Sales from previous week
data['Sales_Lag_1'] = data['Sales'].shift(1)

# Rolling mean for 7-day window
data['Sales_Rolling_Mean'] = data['Sales'].rolling(window=7).mean()

# One-hot encoding for product category (if applicable)
data = pd.get_dummies(data, columns=['Product_Category'], drop_first=True)

# Drop rows with missing target variable (Sales)
data.dropna(subset=['Sales'], inplace=True)
# Export the processed data to a CSV file
data.to_csv('preprocessed_sales_data.csv', index=False)

# Inspect the cleaned data
print(data.head())

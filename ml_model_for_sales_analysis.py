import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the preprocessed data
data = pd.read_csv('preprocessed_sales_data.csv')

# Feature selection - Drop 'Sales' (target) and 'Date' (not needed)
X = data.drop(columns=['Sales', 'Date'])
y = data['Sales']

# Check initial missing values
print("Initial missing values in X:\n", X.isnull().sum())
print("Initial missing values in y:\n", y.isnull().sum())

# Impute missing values for numeric columns with their mean
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())

# Impute missing values for categorical columns with their mode
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# Apply one-hot encoding for categorical columns
if 'Region' in X.columns:
    X = pd.get_dummies(X, columns=['Region'], drop_first=True)

# Check for missing values after preprocessing
print("Missing values in X before splitting:\n", X.isnull().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop columns with missing values in training and testing sets
for dataset, name in [(X_train, "X_train"), (X_test, "X_test")]:
    # Identify columns with missing values
    columns_with_nan = dataset.columns[dataset.isnull().any()]
    # Drop those columns
    dataset.drop(columns=columns_with_nan, inplace=True)
    print(f"Columns dropped in {name} due to missing values: {columns_with_nan.tolist()}")
    print(f"Remaining columns in {name}:\n{dataset.columns.tolist()}")

# Verify that there are no missing values in X_train and X_test
print("Missing values in X_train after dropping columns:\n", X_train.isnull().sum())
print("Missing values in X_test after dropping columns:\n", X_test.isnull().sum())

# Ensure y_train and y_test have no missing values
y_train = y_train.fillna(y_train.mean())
y_test = y_test.fillna(y_test.mean())

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the sales using the test data
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared: {r2}')

# Save evaluation metrics to a DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R-squared'],
    'Value': [rmse, mae, r2]
})

# Combine predictions and actual sales into a DataFrame
predictions_df = pd.DataFrame({
    'Actual Sales': y_test.reset_index(drop=True),
    'Predicted Sales': y_pred
})

# Export both DataFrames to an Excel file with multiple sheets
with pd.ExcelWriter('sales_analysis_results.xlsx', engine='xlsxwriter') as writer:
    metrics_df.to_excel(writer, sheet_name='Evaluation Metrics', index=False)
    predictions_df.to_excel(writer, sheet_name='Predictions vs Actuals', index=False)

print("Results exported to 'sales_analysis_results.xlsx'")

# Plotting the predictions vs actual sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Sales', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Sales', color='red', linestyle='dashed')
plt.legend()
plt.title('Sales Prediction vs Actual Sales')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.show()

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the boston housing dataset
boston_data = load_boston()

# Create a dataframe from the boston_data
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

# Add the target column to the dataframe
df["MEDV"] = boston_data.target

# Select feature and target columns
X = df[["RM"]].values
y = df["MEDV"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Use the model to predict the price of a house with 1000 sq.ft area
house_area = 1000
price = model.predict(np.array([[house_area]]))

# Create a dictionary to store the results
results = {"Mean Squared Error": mse, "R-Squared Value": r2, "Predicted Price of a House": price[0]}

# Create a dataframe from the results dictionary
results_df = pd.DataFrame(results, index=[0])

# Export the dataframe to a CSV file
results_df.to_csv("results.csv", index=False)

# Print the results to the console
print("Mean Squared Error: ", mse)
print("R-Squared Value: ", r2)
print(f"The predicted price of a house with {house_area} sq.ft area is ${price[0]:.2f}")


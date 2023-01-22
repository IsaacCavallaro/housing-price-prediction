import unittest
import numpy as np
import pandas as pd

from housing_price_prediction import load_boston, LinearRegression, mean_squared_error, r2_score, train_test_split

class TestLinearRegression(unittest.TestCase):
    def test_mse(self):
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

        # Assert that the mean squared error is less than a certain value
        self.assertLess(mse, 100)

    def test_r2(self):
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
        r2 = r2_score(y_test, y_pred)

        # Assert that the R-squared value is greater than a certain value
        self.assertGreater(r2, 0.3)



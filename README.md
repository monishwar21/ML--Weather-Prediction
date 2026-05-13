# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries:
2. Load the weather dataset using pandas.
3. Convert categorical data if necessary
4. Select the input features (X) and target variable (y):
5. Split the dataset into training and testing sets using train_test_split().
6. Create the Random Forest model using RandomForestClassifier() or RandomForestRegressor() based on prediction type.
7. Train the model using the fit() method with training data.
8. Predict the weather using the predict() method on test data.
9. Compare predicted results with actual weather values.
10. Visualize the prediction results using graphs if required.

## Program:
```

Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: K MONISHWAR
RegisterNumber:  212225230188
#CODING
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Downloads\weather.CSV.csv")


X = data[['hum', 'pressure', 'wind_speed', 'illumination', 'co2']]

X = X.fillna(X.mean())

y_pollution = data['pm2_5'].fillna(data['pm2_5'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pollution, test_size=0.2, random_state=42
)

pollution_model = DecisionTreeRegressor(random_state=42, max_depth=5)
pollution_model.fit(X_train, y_train)

pollution_pred = pollution_model.predict(X_test)

rmse_pollution = np.sqrt(mean_squared_error(y_test, pollution_pred))
r2_pollution = r2_score(y_test, pollution_pred)
accuracy_pollution = r2_pollution * 100

print(" Pollution Prediction (PM2.5)")
print("Accuracy (%):", accuracy_pollution)

print("R2 Score:", r2_pollution)


print("RMSE:", rmse_pollution)
print("R2 Score:", r2_pollution)

y_temp = data['tem'].fillna(data['tem'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42
)

temp_model = DecisionTreeRegressor(random_state=42, max_depth=5)
temp_model.fit(X_train, y_train)

temp_pred = temp_model.predict(X_test)

rmse_temp = np.sqrt(mean_squared_error(y_test, temp_pred))
r2_temp = r2_score(y_test, temp_pred)
accuracy_temp = r2_temp * 100
print("\n Temperature Prediction")
print("Accuracy (%):", accuracy_temp)
print("RMSE:", rmse_temp)
print("R2 Score:", r2_temp)


y_energy = data['tsr'].fillna(data['tsr'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_energy, test_size=0.2, random_state=42
)

energy_model = DecisionTreeRegressor(random_state=42, max_depth=5)
energy_model.fit(X_train, y_train)

energy_pred = energy_model.predict(X_test)

rmse_energy = np.sqrt(mean_squared_error(y_test, energy_pred))
r2_energy = r2_score(y_test, energy_pred)
accuracy_energy = r2_energy * 100
print("\n Energy Prediction (TSR)")
print("Accuracy (%):", accuracy_energy)


print("RMSE:", rmse_energy)
print("R2 Score:", r2_energy)
```

## Output:

<img width="1042" height="567" alt="image" src="https://github.com/user-attachments/assets/674a872a-5b42-4294-b41d-79813278f548" />

## Result:
The Random Forest Algorithm model for Weather Prediction was implemented successfully, and the weather conditions were predicted with good accuracy using the Random Forest technique.

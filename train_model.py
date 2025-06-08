
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_excel("Data_Train.xlsx")

# Data preprocessing
df.dropna(inplace=True)

# Date of Journey
df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
df.drop(["Date_of_Journey"], axis=1, inplace=True)

# Departure and Arrival time
df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
df.drop(["Dep_Time"], axis=1, inplace=True)

df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
df.drop(["Arrival_Time"], axis=1, inplace=True)

# Duration
df["Duration"] = df["Duration"].apply(lambda x: x.replace("h", "h ").replace("m", "m"))
duration = df["Duration"].str.extract(r'(?:(\d+)h)?\s?(?:(\d+)m)?')
df["Duration_hours"] = duration[0].fillna(0).astype(int)
df["Duration_mins"] = duration[1].fillna(0).astype(int)
df.drop(["Duration"], axis=1, inplace=True)

# Categorical: Total Stops
df["Total_Stops"] = df["Total_Stops"].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2,
                                                '3 stops': 3, '4 stops': 4}).astype(int)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["Airline", "Source", "Destination", "Additional_Info"], drop_first=True)

# Drop Route and build model
df.drop("Route", axis=1, inplace=True)

# Features and Target
X = df.drop(["Price"], axis=1)
y = df["Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")

# Save model
with open("flight_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save features for use in Streamlit
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

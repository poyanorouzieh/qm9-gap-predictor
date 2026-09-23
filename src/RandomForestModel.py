import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# add dataset to file
file_path = pathlib.Path(__file__).resolve()
data_file = file_path.parent.parent / "data" / "qm9_dataset_full_data_v1.csv"

# analyzing the data with pandas in a data frame
df = pd.read_csv(data_file)

# bec dataset is to big and take longer time to analysis
df = df.sample(frac=0.5, random_state=42)

y = df["gap"]

# deleting data that is useless
drop_cols = ["source_file", "index", "na", "homo", "lumo", "gap", "U", "H", "G"]
X = df.drop(columns=drop_cols)

# splitting the data into test and training set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# use regression model
print("learning...")

model = make_pipeline(
    RandomForestRegressor()
)

# train the model
model.fit(X_train, y_train)

# test model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae}")
print(f"RMSE: {rmse}")
print(f"R²:   {r2}")


save_path = pathlib.Path(__file__).parent.parent / "models" / "RandomForestModel.pkl"

joblib.dump(model,save_path)
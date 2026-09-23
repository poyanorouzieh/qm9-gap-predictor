import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# add dataset to file
file_path = pathlib.Path(__file__).resolve()
data_file = file_path.parent.parent / "dataset" / "qm9_dataset_full_data_v1.csv"

# analyzing the data with pandas in a data frame
df = pd.read_csv(data_file)
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
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1)
)

model.fit(X_train, y_train)


# print intercept
print("Intercept:", model.named_steps["ridge"].intercept_)

# idk --> zip ??
feature_names = model.named_steps["polynomialfeatures"].get_feature_names_out(
    X_train.columns
)
print(f"Number of features: {len(feature_names)}")

print("Coefficients:")
for name, coef in zip(feature_names, model.named_steps["ridge"].coef_):
    print(f"  {name}: {coef}")

# model quality assessment
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae}")
print(f"RMSE: {rmse}")
print(f"R²:   {r2}")

save_path = pathlib.Path(__file__).parent.parent / "models"/ "models" / "LinearRegression(Ridge_alpha=1,degree=3,R=0.5968).pkl"

joblib.dump(model, save_path)

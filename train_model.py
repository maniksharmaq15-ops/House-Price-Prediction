# Train the house price model and save it for the web app
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("Mumbai_House_Prices_Cleaned.csv")

cat_cols = ["type", "region", "status", "age"]
num_cols = ["bhk", "area"]
X = df[num_cols + cat_cols]
y = df["price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

model = Pipeline([
    ("prep", prep),
    ("rf", RandomForestRegressor(n_estimators=60, min_samples_leaf=5, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
print("R2:", round(r2_score(y_test, pred), 3))
print("MAE (Lakhs):", round(mean_absolute_error(y_test, pred), 2))

# options for the dropdowns in the web page
options = {c: sorted(df[c].unique().tolist()) for c in cat_cols}
# only keep regions that have at least 20 listings
counts = df["region"].value_counts()
options["region"] = sorted(counts[counts >= 20].index.tolist())
pd.Series(options).to_json("options.json")

joblib.dump(model, "house_price_model.joblib", compress=3)
print("Saved")

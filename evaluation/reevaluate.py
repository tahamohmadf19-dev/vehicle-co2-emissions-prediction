"""Re-evaluate the CO2 model without duplicate leakage.

    python evaluation/reevaluate.py "path/to/CO2 Emissions_Canada.csv"

The dataset is not stored in this repository. The notebook downloads it with kagglehub
("debajyotipodder/co2-emission-by-vehicles"); pass the downloaded CSV path, or set CO2_CSV.

Four evaluations, all with the notebook's features and RandomForest(n_estimators=400, random_state=42):
1. reproduction: the notebook's own pipeline, duplicates kept, random 60/20/20 split
2. dedup_random: exact duplicate rows removed first, same random split
3. dedup_grouped: duplicates removed, 5-fold GroupKFold by (Make, Model), so a vehicle model never
   appears in both training and test
4. baselines under the same grouped folds: a straight line on combined fuel consumption alone, and
   fuel consumption with fuel type
"""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "CO2 Emissions(g/km)"
FUEL_COMB = "Fuel Consumption Comb (L/100 km)"


def features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """The notebook's feature engineering (cell 7), unchanged."""
    out = df.copy()
    out["engine_per_cylinder"] = out["Engine Size(L)"] / out["Cylinders"]
    out["city_hwy_ratio"] = out["Fuel Consumption City (L/100 km)"] / out["Fuel Consumption Hwy (L/100 km)"]
    out = out.drop(columns=["Engine Size(L)", "Cylinders", "Fuel Consumption City (L/100 km)",
                            "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (mpg)"])
    return out.drop(columns=[TARGET]), out[TARGET]


def forest(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    return Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))])


def scores(y_true, y_pred) -> dict:
    return {"r2": round(float(r2_score(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 2)}


def random_split_test(df: pd.DataFrame) -> dict:
    X, y = features(df)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    model = forest(X_train).fit(X_train, y_train)
    test_rows = set(map(tuple, X_test.assign(_y=y_test).values.tolist()))
    train_rows = set(map(tuple, X_train.assign(_y=y_train).values.tolist()))
    return {**scores(y_test, model.predict(X_test)), "rows": len(df), "test_rows": len(X_test),
            "test_rows_identical_to_a_training_row": len(test_rows & train_rows)}


def grouped_cv(df: pd.DataFrame, make_model) -> dict:
    groups = df["Make"].astype(str) + " | " + df["Model"].astype(str)
    folds = []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(df, groups=groups):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        model, X_train, y_train, X_test, y_test = make_model(train, test)
        folds.append(scores(y_test, model.fit(X_train, y_train).predict(X_test)))
    summary = {k: {"mean": round(float(np.mean([f[k] for f in folds])), 4 if k == "r2" else 2),
                   "std": round(float(np.std([f[k] for f in folds])), 4 if k == "r2" else 2)} for k in folds[0]}
    return {**summary, "folds": 5, "groups": int(groups.nunique()), "per_fold": folds}


def forest_case(train, test):
    X_train, y_train = features(train)
    X_test, y_test = features(test)
    return forest(X_train), X_train, y_train, X_test, y_test


def fuel_only(train, test):
    return LinearRegression(), train[[FUEL_COMB]], train[TARGET], test[[FUEL_COMB]], test[TARGET]


def fuel_and_type(train, test):
    cols = [FUEL_COMB, "Fuel Type"]
    pre = ColumnTransformer([("num", "passthrough", [FUEL_COMB]),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Fuel Type"])])
    return Pipeline([("pre", pre), ("model", LinearRegression())]), train[cols], train[TARGET], test[cols], test[TARGET]


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CO2_CSV", ""))
    if not path.is_file():
        print(__doc__)
        return 2
    df = pd.read_csv(path)
    dedup = df.drop_duplicates().reset_index(drop=True)
    results = {
        "dataset": {"rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum()), "rows_after_dedup": len(dedup)},
        "reproduction": random_split_test(df),
        "dedup_random": random_split_test(dedup),
        "dedup_grouped_by_make_model": grouped_cv(dedup, forest_case),
        "baseline_fuel_consumption_only": grouped_cv(dedup, fuel_only),
        "baseline_fuel_consumption_and_fuel_type": grouped_cv(dedup, fuel_and_type),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
    }
    out = Path(__file__).with_name("reevaluation_results.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "environment"}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

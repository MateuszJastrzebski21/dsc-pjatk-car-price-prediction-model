import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

print("Wczytuję dane z pliku CSV (train)...")
df = pd.read_csv("data/cleaned_sales_ads_train_20250317_173818.csv")

# Updated numeric columns
num_cols = ["Cena", "Rok_produkcji", "Przebieg_km", "Moc_KM", "Pojemnosc_cm3", "Liczba_drzwi", "Wiek_pojazdu"]

# Updated categorical columns
cat_cols = ["Waluta", "Stan", "Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu", "Generacja_pojazdu", 
           "Rodzaj_paliwa", "Naped", "Skrzynia_biegow", "Typ_nadwozia", "Kraj_pochodzenia", "Pierwszy_wlasciciel"]

num_cols = [col for col in num_cols if col in df.columns]
cat_cols = [col for col in cat_cols if col in df.columns]
all_cols = num_cols + cat_cols

X = df[all_cols].copy()
y = df["Cena"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", TargetEncoder())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

X_train_transformed = preprocessor.fit_transform(X_train, y_train)
X_test_transformed = preprocessor.transform(X_test)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, step=1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "random_state": 42,
        "tree_method": "gpu_hist" if xgb.__version__ >= "1.0.0" else "hist",
    }

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train_transformed, y_train, scoring="neg_root_mean_squared_error", cv=5)
    return -scores.mean()

print("Optymalizacja hiperparametrów za pomocą Optuny...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Najlepsze parametry znalezione przez Optunę:", best_params)

xgb_model = xgb.XGBRegressor(**best_params)

print("Przeprowadzam walidację krzyżową dla finalnego modelu (CV=5)...")
cv_scores = cross_val_score(xgb_model, X_train_transformed, y_train, scoring="neg_root_mean_squared_error", cv=5)
mean_cv_rmse = -cv_scores.mean()
print(f"Średni RMSE w walidacji krzyżowej: {mean_cv_rmse:.2f}")

print("Trenuję model XGBRegressor...")
xgb_model.fit(
    X_train_transformed,
    y_train,
    eval_set=[(X_test_transformed, y_test)],
    early_stopping_rounds=100,
    verbose=True
)

y_pred = xgb_model.predict(X_test_transformed)
mae_value = mean_absolute_error(y_test, y_pred)
rmse_value = np.sqrt(mean_squared_error(y_test, y_pred))
r2_value = r2_score(y_test, y_pred)

print(f"MAE = {mae_value:.3f}, RMSE = {rmse_value:.3f}, R2 = {r2_value:.3f}")

importance = xgb_model.get_booster().get_score(importance_type="gain")
importance_df = pd.DataFrame({
    "Cecha": list(importance.keys()),
    "Waga": list(importance.values())
}).sort_values(by="Waga", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Cecha"][:20], importance_df["Waga"][:20])
plt.gca().invert_yaxis()
plt.xlabel("Ważność")
plt.ylabel("Cecha")
plt.title("Top 20 cech wg ważności w XGBoost")
plt.show()

importance_df.to_csv("feature_importance.csv", index=False)
print("Zapisano feature_importance.csv")

print("Wczytuję dane testowe (sales_ads_test.csv)...")
test_df = pd.read_csv("data/sales_ads_test.csv")

test_num_cols = [col for col in num_cols if col in test_df.columns]
test_cat_cols = [col for col in cat_cols if col in test_df.columns]
test_all_cols = test_num_cols + test_cat_cols

X_submission = test_df[test_all_cols].copy()

print("Przetwarzam zbiór testowy (transformuję) ...")
X_submission_transformed = preprocessor.transform(X_submission)

print("Przewiduję ceny...")
y_submission_pred = xgb_model.predict(X_submission_transformed)

y_submission_pred = np.round(y_submission_pred).astype(int)

submission_df = pd.DataFrame({
    "ID": test_df["ID"],
    "Cena": y_submission_pred
})

submission_df.to_csv("submission.csv", index=False)
print("Zapisano submission.csv – plik gotowy do wysyłki.")



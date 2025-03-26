import os
import ast
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
from category_encoders import CatBoostEncoder

EXCHANGE_RATE_EUR_TO_PLN = 4.5 

# Pomocnicze funkcje do parsowania i multi-hot kolumny "Wyposazenie"

def parse_equipment_inplace(df, col="Wyposazenie"):
    if col not in df.columns:
        return
    df[col] = df[col].fillna("[]")
    def try_eval(x):
        try:
            if isinstance(x, str):
                return ast.literal_eval(x)
            else:
                return x
        except:
            return []
    df[col] = df[col].apply(try_eval)

def get_all_equipment_values(*dfs, col="Wyposazenie"):
    all_equip = set()
    for df in dfs:
        if col in df.columns:
            for val in df[col].dropna():
                if isinstance(val, list):
                    for item in val:
                        item = item.strip()
                        all_equip.add(item)
    return all_equip

def add_equipment_features(df, all_equip, col="Wyposazenie"):
    if col not in df.columns:
        return df
    for eq_item in all_equip:
        new_col = "eq_" + eq_item.replace(" ", "_").replace("/", "_")
        df[new_col] = 0
    for i, val in df[col].items():
        if isinstance(val, list):
            for item in val:
                item = item.strip()
                new_col = "eq_" + item.replace(" ", "_").replace("/", "_")
                if new_col in df.columns:
                    df.at[i, new_col] = 1
    return df

def create_location_embedding(df_train, df_test, location_col="Lokalizacja"):
    """
    Tworzy embedding dla lokalizacji na podstawie średnich cen w zbiorze treningowym.
    Zwraca DataFrame z embeddingami dla train i test.
    """
    location_means = df_train.groupby(location_col)["Cena"].mean().to_dict()
    
    default_mean = df_train["Cena"].mean()
    
    for df in [df_train, df_test]:
        if location_col in df.columns:
            df[f"loc_embedding_mean_price"] = df[location_col].map(lambda x: location_means.get(x, default_mean))
        else:
            df[f"loc_embedding_mean_price"] = default_mean  # Jeśli brak kolumny, wstawiamy średnią
    
    return df_train, df_test

# DataLoader
class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        print("Wczytuję dane z pliku CSV (train)...")
        df = pd.read_csv(self.train_path)
        df["Cena"] = df["Cena"].astype(float)
        df = self.convert_prices_to_pln(df)
        if "Waluta" in df.columns:
            df["Waluta"] = df["Waluta"].fillna("PLN")
        for col in [
            "Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu", "Generacja_pojazdu",
            "Naped", "Skrzynia_biegow", "Kraj_pochodzenia", "Lokalizacja"
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("missing")
        parse_equipment_inplace(df, col="Wyposazenie")
        return df

    def load_test_data(self):
        print("Wczytuję dane testowe...")
        df = pd.read_csv(self.test_path)
        df["Is_EUR"] = df["Waluta"] == "EUR"
        if "Waluta" in df.columns:
            df["Waluta"] = df["Waluta"].fillna("PLN")
        for col in [
            "Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu", "Generacja_pojazdu",
            "Naped", "Skrzynia_biegow", "Kraj_pochodzenia", "Lokalizacja"
        ]:
            if col in df.columns:
                df[col] = df[col].fillna("missing")
        parse_equipment_inplace(df, col="Wyposazenie")
        return df

    @staticmethod
    def convert_prices_to_pln(df):
        mask_eur = df["Waluta"] == "EUR"
        df["Cena"] = df["Cena"].astype(float)
        df.loc[mask_eur, "Cena"] = df.loc[mask_eur, "Cena"] * EXCHANGE_RATE_EUR_TO_PLN
        df["Waluta"] = "PLN"
        return df

# Preprocessor (z dodaną obsługą embeddingów lokalizacji)
class Preprocessor:
    def __init__(self, df):
        self.num_cols = [
            "Rok_produkcji", "Przebieg_km", "Moc_KM", "Pojemnosc_cm3",
            "Emisja_CO2", "Liczba_drzwi", "loc_embedding_mean_price"  # Dodajemy embedding
        ]
        self.cat_cols = [
            "Waluta", "Stan", "Marka_pojazdu", "Model_pojazdu", "Rodzaj_paliwa",
            "Typ_nadwozia", "Kolor", "Pierwszy_wlasciciel", "Wersja_pojazdu",
            "Generacja_pojazdu", "Naped", "Skrzynia_biegow", "Kraj_pochodzenia",
            "Lokalizacja" 
        ]
        eq_cols = [c for c in df.columns if c.startswith("eq_")]
        self.num_cols += eq_cols
        self.num_cols = [c for c in self.num_cols if c in df.columns]
        self.cat_cols = [c for c in self.cat_cols if c in df.columns]

        self.num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        self.cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("catboost_enc", CatBoostEncoder())
        ])
        self.preprocessor = ColumnTransformer([
            ("num", self.num_pipeline, self.num_cols),
            ("cat", self.cat_pipeline, self.cat_cols)
        ])
        self.final_feature_names_ = self.num_cols + self.cat_cols

    def transform(self, X, y=None):
        if y is not None:
            return self.preprocessor.fit_transform(X, y)
        return self.preprocessor.transform(X)

# ModelTrainer (bez zmian)
class ModelTrainer:
    def __init__(self, params, feature_names=None):
        self.model = xgb.XGBRegressor(**params)
        self.feature_names = feature_names
        self.is_fitted = False
    
    def cross_validate(self, X, y):
        print("Przeprowadzam walidację krzyżową (oryginalna skala)...")
        cv_scores = cross_val_score(self.model, X, y, scoring="neg_root_mean_squared_error", cv=5)
        print(f"Średni RMSE (oryginalna skala) w walidacji krzyżowej: {-cv_scores.mean():.4f}")
    
    def train(self, X, y):
        print("Trenuję model XGBRegressor (oryginalna skala targetu)...")
        self.model.fit(X, y)
        self.is_fitted = True
    
    def evaluate(self, X, y):
        pred = self.model.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
        print(f"[Oryg. skala] MAE = {mae:.3f}, RMSE = {rmse:.3f}, R2 = {r2:.3f}")
    
    def predict(self, X):
        pred = self.model.predict(X)
        return np.round(pred).astype(int)
    
    def feature_importance(self, output_path="importance.csv"):
        if not self.is_fitted:
            print("Model nie jest jeszcze wytrenowany, brak importances.")
            return None
        booster = self.model.get_booster()
        importance_dict = booster.get_score(importance_type="gain")
        items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        feature_indices = [int(k.replace("f", "")) for k, v in items]
        gains = [v for k, v in items]
        df_imp = pd.DataFrame({"FeatureIndex": feature_indices, "Gain": gains})
        if self.feature_names is not None and len(self.feature_names) > 0:
            df_imp["FeatureName"] = df_imp["FeatureIndex"].apply(
                lambda i: self.feature_names[i] if i < len(self.feature_names) else f"???_{i}"
            )
        else:
            df_imp["FeatureName"] = df_imp["FeatureIndex"].apply(lambda i: f"f{i}")
        df_imp = df_imp[["FeatureIndex", "FeatureName", "Gain"]].sort_values(by="Gain", ascending=False)
        df_imp.to_csv(output_path, index=False)
        print(f"Zapisano importance do pliku: {output_path}")
        return df_imp

# SubmissionHandler (bez zmian)
class SubmissionHandler:
    def __init__(self, output_folder="tester"):
        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder
    
    def save_submission(self, test_df, predictions, filename="submission.csv"):
        test_df["Cena"] = predictions
        mask_eur = test_df["Is_EUR"]
        test_df.loc[mask_eur, "Cena"] = test_df.loc[mask_eur, "Cena"] / EXCHANGE_RATE_EUR_TO_PLN
        test_df["Cena"] = test_df["Cena"].round(2)
        submission_df = test_df[["ID", "Cena"]]
        path = os.path.join(self.output_folder, filename)
        submission_df.to_csv(path, index=False)
        print(f"Zapisano {path} – plik gotowy do wysyłki.")

# Główny skrypt
if __name__ == "__main__":
    data_loader = DataLoader(
        "../input/dane-dla-dsc/sales_ads_train.csv",
        "../input/dane-dla-dsc/sales_ads_test.csv"
    )
    df_train = data_loader.load_data()
    df_test = data_loader.load_test_data()

    all_equip = get_all_equipment_values(df_train, df_test, col="Wyposazenie")

    df_train = add_equipment_features(df_train, all_equip, col="Wyposazenie")
    df_test = add_equipment_features(df_test, all_equip, col="Wyposazenie")

    if "Wyposazenie" in df_train.columns:
        df_train.drop(columns=["Wyposazenie"], inplace=True)
    if "Wyposazenie" in df_test.columns:
        df_test.drop(columns=["Wyposazenie"], inplace=True)

    df_train, df_test = create_location_embedding(df_train, df_test, location_col="Lokalizacja")  # lub "Kraj_pochodzenia"

    preprocessor = Preprocessor(df_train)
    X = df_train[preprocessor.num_cols + preprocessor.cat_cols].copy()
    y = df_train["Cena"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_transformed = preprocessor.transform(X_train, y_train)
    X_val_transformed = preprocessor.transform(X_val)

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
            "device": "cuda",
            "tree_method": "hist",
            "predictor": "gpu_predictor"
        }
        model = xgb.XGBRegressor(**params)
        cv_scores = cross_val_score(model, X_train_transformed, y_train, scoring="neg_root_mean_squared_error", cv=5)
        return -cv_scores.mean()

    print("### Optymalizacja hiperparametrów za pomocą Optuny... ###")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print("Najlepsze parametry znalezione przez Optunę:", best_params)
    best_params["random_state"] = 42
    best_params["device"] = "cuda"
    best_params["tree_method"] = "hist"
    best_params["predictor"] = "gpu_predictor"

    final_model_trainer = ModelTrainer(best_params, feature_names=preprocessor.final_feature_names_)
    final_model_trainer.cross_validate(X_train_transformed, y_train)
    final_model_trainer.train(X_train_transformed, y_train)

    print("Wyniki na zbiorze walidacyjnym (oryginalna skala):")
    final_model_trainer.evaluate(X_val_transformed, y_val)

    importance_df = final_model_trainer.feature_importance(output_path="importance.csv")
    print(importance_df.head(25))

    X_test = df_test[preprocessor.num_cols + preprocessor.cat_cols].copy()
    X_test_transformed = preprocessor.transform(X_test)
    y_test_pred = final_model_trainer.predict(X_test_transformed)

    submission_handler = SubmissionHandler()
    submission_handler.save_submission(df_test, y_test_pred, filename="submission.csv")
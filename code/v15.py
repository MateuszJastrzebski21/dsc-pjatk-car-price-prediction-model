import os
import ast
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import CatBoostEncoder
from sklearn.cluster import KMeans

EXCHANGE_RATE_EUR_TO_PLN = 4.5

def determine_price_segments(df, price_col="Cena", n_segments=9):
    """
    Wyznacza progi dla segmentów na podstawie logarytmu cen za pomocą k-means (9 klastrów).
    Zwraca DataFrame z dodaną kolumną Segment oraz progi między segmentami.
    """
    prices = df[price_col].dropna().values
    log_prices = np.log1p(prices).reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_segments, random_state=42)
    kmeans.fit(log_prices)

    centers = np.sort(kmeans.cluster_centers_.flatten())
    boundaries = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]
    boundaries = [np.expm1(boundary) for boundary in boundaries]

    def assign_segment(price):
        if pd.isna(price):
            return "nieznane"
        for i, boundary in enumerate(boundaries):
            if price <= boundary:
                return f"segment_{i+1}"
        return f"segment_{n_segments}"

    df["Segment"] = df[price_col].apply(assign_segment)
    return df, boundaries

def assign_segment_for_new_data(price, boundaries):
    if pd.isna(price):
        return "nieznane"
    for i, boundary in enumerate(boundaries):
        if price <= boundary:
            return f"segment_{i+1}"
    return f"segment_9"

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

class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def clip_outliers(self, df, col="Cena"):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

    def add_feature_interactions(self, df):
        if "Rok_produkcji" in df.columns and "Przebieg_km" in df.columns:
            df["Rok_x_Przebieg"] = df["Rok_produkcji"] * df["Przebieg_km"]
        if "Moc_KM" in df.columns and "Pojemnosc_cm3" in df.columns:
            df["Moc_x_Pojemnosc"] = df["Moc_KM"] * df["Pojemnosc_cm3"]
        return df

    def load_data(self):
        print("Wczytuję dane z pliku CSV (train)...")
        df = pd.read_csv(self.train_path)
        df["Cena"] = df["Cena"].astype(float)
        df = self.convert_prices_to_pln(df)
        df = self.clip_outliers(df)
        df = self.add_feature_interactions(df)

        if "Waluta" in df.columns:
            df["Waluta"] = df["Waluta"].fillna("PLN")

        for col in ["Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu", "Generacja_pojazdu", "Naped", "Skrzynia_biegow", "Kraj_pochodzenia"]:
            if col in df.columns:
                df[col] = df[col].fillna("missing")

        parse_equipment_inplace(df)
        df, self.price_boundaries = determine_price_segments(df)
        print(f"Wyznaczone progi między segmentami: {self.price_boundaries}")
        return df

    def load_test_data(self):
        print("Wczytuję dane testowe...")
        df = pd.read_csv(self.test_path)
        df["Is_EUR"] = df["Waluta"] == "EUR"
        df = self.add_feature_interactions(df)

        if "Waluta" in df.columns:
            df["Waluta"] = df["Waluta"].fillna("PLN")

        for col in ["Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu", "Generacja_pojazdu", "Naped", "Skrzynia_biegow", "Kraj_pochodzenia"]:
            if col in df.columns:
                df[col] = df[col].fillna("missing")

        parse_equipment_inplace(df)
        df["Segment"] = "nieznane"
        return df

    @staticmethod
    def convert_prices_to_pln(df):
        mask_eur = df["Waluta"] == "EUR"
        df["Cena"] = df["Cena"].astype(float)
        df.loc[mask_eur, "Cena"] = df.loc[mask_eur, "Cena"] * EXCHANGE_RATE_EUR_TO_PLN
        df["Waluta"] = "PLN"
        return df

class Preprocessor:
    def __init__(self, df):
        self.num_cols = [
            "Rok_produkcji", "Przebieg_km", "Moc_KM", "Pojemnosc_cm3", "Emisja_CO2", "Liczba_drzwi",
            "Rok_x_Przebieg", "Moc_x_Pojemnosc"
        ]
        self.cat_cols = [
            "Waluta", "Stan", "Marka_pojazdu", "Model_pojazdu", "Rodzaj_paliwa", "Typ_nadwozia",
            "Kolor", "Pierwszy_wlasciciel", "Wersja_pojazdu", "Generacja_pojazdu", "Naped",
            "Skrzynia_biegow", "Kraj_pochodzenia"
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


class ModelTrainer:
    def __init__(self, params, feature_names=None):
        self.models = {}
        self.params = params
        self.feature_names = feature_names
        self.is_fitted = False

    def train(self, X, y, segments):
        print("Trenuję modele dla każdego segmentu z log-transformacją...")
        self.segments = segments.unique()
        for segment in self.segments:
            if segment == "nieznane":
                continue
            mask = segments == segment
            if mask.sum() > 0:
                y_log = np.log1p(y[mask])
                weights = y[mask] / y[mask].mean() 
                model = xgb.XGBRegressor(**self.params)
                model.fit(X[mask], y_log, sample_weight=weights)
                self.models[segment] = model
        self.is_fitted = True

    def cross_validate(self, X, y, segments):
        print("Przeprowadzam stratyfikowaną walidację krzyżową dla każdego segmentu...")
        for segment in segments.unique():
            if segment == "nieznane":
                continue
            mask = segments == segment
            if mask.sum() > 0:
                X_seg = X[mask]
                y_seg = y[mask]
                y_log_seg = np.log1p(y_seg)
                weights_seg = y_seg / y_seg.mean()

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = []
                for train_idx, val_idx in skf.split(X_seg, pd.cut(y_seg, bins=5, labels=False)):
                    X_train_fold, X_val_fold = X_seg[train_idx], X_seg[val_idx]
                    y_train_fold, y_val_fold = y_log_seg[train_idx], y_seg[val_idx]
                    weights_fold = weights_seg[train_idx]

                    model = xgb.XGBRegressor(**self.params)
                    model.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
                    y_pred_log = model.predict(X_val_fold)
                    y_pred = np.expm1(y_pred_log)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                    cv_scores.append(rmse)

                mean_rmse = np.mean(cv_scores)
                print(f"Segment {segment}: Średni RMSE = {mean_rmse:.4f} (stratyfikowana walidacja)")

    def evaluate(self, X, y, segments):
        print("Ewaluacja dla każdego segmentu...")
        for segment in segments.unique():
            if segment == "nieznane":
                continue
            mask = segments == segment
            if mask.sum() > 0 and segment in self.models:
                pred_log = self.models[segment].predict(X[mask])
                pred = np.expm1(pred_log)
                mae = mean_absolute_error(y[mask], pred)
                rmse = np.sqrt(mean_squared_error(y[mask], pred))
                r2 = r2_score(y[mask], pred)
                print(f"Segment {segment}: MAE = {mae:.3f}, RMSE = {rmse:.3f}, R2 = {r2:.3f}")

    def predict(self, X, segments):
        predictions = np.zeros(len(X))
        for i, segment in enumerate(segments):
            if segment == "nieznane" or segment not in self.models:
                segment = "segment_1" if "segment_1" in self.models else list(self.models.keys())[0]
            pred_log = self.models[segment].predict(X[i:i+1])[0]
            predictions[i] = np.expm1(pred_log)
        return np.round(predictions).astype(int)

    def feature_importance(self, output_path="importance"):
        if not self.is_fitted:
            print("Modele nie są jeszcze wytrenowane.")
            return None
        for segment in self.models:
            booster = self.models[segment].get_booster()
            importance_dict = booster.get_score(importance_type="gain")
            items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            feature_indices = []
            gains = []
            for k, v in items:
                idx_str = k.replace("f", "")
                idx = int(idx_str)
                feature_indices.append(idx)
                gains.append(v)

            df_imp = pd.DataFrame({"FeatureIndex": feature_indices, "Gain": gains})
            if self.feature_names and len(self.feature_names) > 0:
                df_imp["FeatureName"] = df_imp["FeatureIndex"].apply(
                    lambda i: self.feature_names[i] if i < len(self.feature_names) else f"???_{i}"
                )
            else:
                df_imp["FeatureName"] = df_imp["FeatureIndex"].apply(lambda i: f"f{i}")

            df_imp = df_imp[["FeatureIndex", "FeatureName", "Gain"]].sort_values(by="Gain", ascending=False)
            path = f"{output_path}_{segment}.csv"
            df_imp.to_csv(path, index=False)
            print(f"Zapisano importance dla segmentu {segment} do pliku: {path}")

class SubmissionHandler:
    def __init__(self, output_folder="tester"):
        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder

    def save_submission(self, test_df, predictions, boundaries, filename="submission.csv"):
        test_df["Cena"] = predictions
        test_df["Segment"] = test_df["Cena"].apply(lambda x: assign_segment_for_new_data(x, boundaries))
        mask_eur = test_df["Is_EUR"]
        test_df.loc[mask_eur, "Cena"] = test_df.loc[mask_eur, "Cena"] / EXCHANGE_RATE_EUR_TO_PLN
        test_df["Cena"] = test_df["Cena"].round(2)
        submission_df = test_df[["ID", "Cena"]]
        path = os.path.join(self.output_folder, filename)
        submission_df.to_csv(path, index=False)
        print(f"Zapisano {path} – plik gotowy do wysyłki.")


if __name__ == "__main__":
    data_loader = DataLoader("data/sales_ads_train.csv", "data/sales_ads_test.csv")
    df_train = data_loader.load_data()
    df_test = data_loader.load_test_data()
    price_boundaries = data_loader.price_boundaries

    print("\nLiczność samochodów w poszczególnych segmentach (dane treningowe):")
    print(df_train["Segment"].value_counts())

    all_equip = get_all_equipment_values(df_train, df_test)
    df_train = add_equipment_features(df_train, all_equip)
    df_test = add_equipment_features(df_test, all_equip)

    if "Wyposazenie" in df_train.columns:
        df_train.drop(columns=["Wyposazenie"], inplace=True)
    if "Wyposazenie" in df_test.columns:
        df_test.drop(columns=["Wyposazenie"], inplace=True)

    preprocessor = Preprocessor(df_train)
    X = df_train[preprocessor.num_cols + preprocessor.cat_cols].copy()
    y = df_train["Cena"].values
    segments_train = df_train["Segment"]

    X_train, X_val, y_train, y_val, segments_train, segments_val = train_test_split(
        X, y, segments_train, test_size=0.2, random_state=42, stratify=segments_train
    )

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
        y_log = np.log1p(y_train)
        weights = y_train / y_train.mean()
        model.fit(X_train_transformed, y_log, sample_weight=weights)
        y_pred_log = model.predict(X_train_transformed)
        y_pred = np.expm1(y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        return rmse

    print("### Optymalizacja hiperparametrów za pomocą Optuny... ###")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print("Najlepsze parametry znalezione przez Optunę:", best_params)
    best_params["random_state"] = 42
    best_params["device"] = "cuda"
    best_params["tree_method"] = "hist"
    best_params["predictor"] = "gpu_predictor"

    final_model_trainer = ModelTrainer(best_params, feature_names=preprocessor.final_feature_names_)
    final_model_trainer.cross_validate(X_train_transformed, y_train, segments_train)
    final_model_trainer.train(X_train_transformed, y_train, segments_train)

    print("Wyniki na zbiorze walidacyjnym (oryginalna skala):")
    final_model_trainer.evaluate(X_val_transformed, y_val, segments_val)

    final_model_trainer.feature_importance(output_path="importance")

    print("Przygotowuje dane testowe")
    X_test = df_test[preprocessor.num_cols + preprocessor.cat_cols].copy()
    X_test_transformed = preprocessor.transform(X_test)

    print("Wstępne predykcje do określenia segmentów")
    prelim_model = xgb.XGBRegressor(**best_params)
    prelim_model.fit(X_train_transformed, np.log1p(y_train))
    prelim_predictions_log = prelim_model.predict(X_test_transformed)
    prelim_predictions = np.expm1(prelim_predictions_log)
    segments_test = [assign_segment_for_new_data(price, price_boundaries) for price in prelim_predictions]

    print("Końcowe predykcje") 
    y_test_pred = final_model_trainer.predict(X_test_transformed, segments_test)


    print("Zapisanie wyników do pliku submission")
    submission_handler = SubmissionHandler()
    submission_handler.save_submission(df_test, y_test_pred, price_boundaries, filename="submission.csv")
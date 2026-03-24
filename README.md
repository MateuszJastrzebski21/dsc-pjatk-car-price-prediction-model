# Car Price Prediction Model
**DSC PJATK Internal Kaggle Competition**

> Predicting used car prices from real Polish automotive listings data.  
> **Final result: RMSE ~19,500 PLN on private leaderboard.**

---

## Problem

Given real sales listings from a Polish automotive marketplace, predict the price (`Cena`) of a used car. The dataset contained ~100k rows with mixed numeric and categorical features including brand, model, generation, mileage, engine size, fuel type, gearbox, and equipment lists.

Key challenges:
- Heavy right skew in price distribution (range: ~1k to 1M+ PLN)
- ~75k missing values in `Emisja_CO2`, ~3k missing in `Stan` and `Waluta`
- High-cardinality categoricals: `Model_pojazdu`, `Wersja_pojazdu`, `Generacja_pojazdu`
- Equipment column stored as a raw Python list string (`"['ABS', 'ESP', ...]"`)
- Mixed currencies (PLN and EUR)

---

## Iterative Development

### V1 — Baseline
- XGBoost with default params (`n_estimators=100`, `lr=0.1`, `max_depth=6`)
- OneHotEncoder for categoricals, median imputation for numerics
- First submission returned score of 0.0 due to float vs. int formatting issue

### V2 — Format fix
- Predictions rounded and cast to `int` — submission format corrected
- Same model and preprocessing as V1
- Established first real benchmark: **RMSE ~25,000 PLN**

### V3 — Manual hyperparameter tuning + CV
- Manually adjusted params: `n_estimators=500`, `lr=0.05`, `max_depth=4`
- Added 5-fold cross-validation to estimate generalization error
- Slight improvement over V2

### V4 — Optuna + TargetEncoder
- Replaced OneHotEncoder with **TargetEncoder** (better for high-cardinality features)
- Introduced **Optuna** for Bayesian hyperparameter search (20–50 trials)
- Best params found: `n_estimators=3000`, `lr=0.022`, `max_depth=11`
- Result: **RMSE ~21,000 PLN**

### V5 — Custom data cleaning pipeline
- Separated data cleaning into its own module (`data_clean.py`)
- Imputed missing `Rok_produkcji` from vehicle generation strings and mileage proximity
- Imputed missing `Stan` (new/used) from year + mileage heuristics
- Added `Wiek_pojazdu` (vehicle age) as engineered feature
- Dropped low-signal columns: `Emisja_CO2`, `Kolor`, `Lokalizacja_oferty`
- Extended categoricals: added `Generacja_pojazdu`, `Naped`, `Skrzynia_biegow`, `Kraj_pochodzenia`
- Result: overfitting — model regressed due to train/test preprocessing mismatch

### V6–V12 — Architecture refactor + GPU
- Refactored codebase into OOP: `DataLoader`, `Preprocessor`, `ModelTrainer`, `SubmissionHandler`
- Added EUR→PLN currency conversion with back-conversion for EUR-priced test listings
- Switched to `device="cuda"` + `tree_method="hist"` for GPU acceleration
- Experimented with location embedding (mean price per city)
- Iterative Optuna tuning (10–20 trials per run)

### V13 — CatBoostEncoder
- Replaced TargetEncoder with **CatBoostEncoder** — reduces target leakage in CV folds
- Improved handling of `Wersja_pojazdu` and `Generacja_pojazdu`
- Result: RMSE approaching 21,000 PLN

### V14 — Equipment multi-hot encoding
- Parsed `Wyposazenie` column (raw list string) into 100+ binary `eq_*` features (e.g. `eq_ABS`, `eq_ESP`)
- Used `ast.literal_eval` for safe parsing; unified vocabulary across train and test sets
- CatBoostEncoder + expanded feature set + Optuna tuning
- Leaderboard result: **RMSE ~19,500 PLN**

### V15 — Price segmentation via KMeans
- Applied **KMeans clustering** (9 clusters) on log-transformed prices to define price segments
- Trained a separate XGBoost model per segment with log-transformed target and sample weights
- Stratified train/val split by segment; StratifiedKFold cross-validation per segment
- Added feature interactions: `Rok_x_Przebieg`, `Moc_x_Pojemnosc`
- Preliminary pass to assign test set segments before final prediction

---

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | XGBoost, scikit-learn |
| Hyperparameter tuning | Optuna (Bayesian search, up to 50 trials) |
| Encoding | CatBoostEncoder (category_encoders) |
| Clustering | KMeans (scikit-learn) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| GPU | CUDA (`device="cuda"`, `tree_method="hist"`) |
| Environment | Python 3.12, uv |

---

## Repository Structure
```
├── code/                    # Source code versions (V1–V15)
├── data/                    # Raw and cleaned datasets (not included)
├── clustering/              # KMeans analysis scripts
├── export_data_for_model/   # Cleaned train/test exports
└── README.md
```

## Key Learnings

- **TargetEncoder vs CatBoostEncoder**: CatBoostEncoder handles target leakage better in CV by using ordered boosting-style encoding, giving more reliable validation estimates
- **Equipment features mattered most**: Adding 100+ binary `eq_*` flags was the single biggest RMSE improvement, dropping from ~21k to ~19.5k PLN
- **Price segmentation**: Training separate models per price segment reduced error on premium vehicles where a single global model struggled
- **Data cleaning tradeoff**: Aggressive cleaning (V5) introduced train/test mismatch and hurt leaderboard score despite better local CV — a practical lesson in pipeline consistency

---

*Project developed as part of the DSC PJATK recruitment competition, March 2025.*  
*Team: Mateusz Jastrzębski & Damian Ruczyński*

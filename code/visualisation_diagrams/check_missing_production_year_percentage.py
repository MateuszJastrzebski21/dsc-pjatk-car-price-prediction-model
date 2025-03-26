import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt


def check_missing_percentage(df, column="Rok_produkcji"):
    missing_count = df[column].isnull().sum()
    total_count = len(df)
    missing_percentage = (missing_count / total_count) * 100
    print(f"Procent braków w '{column}': {missing_percentage:.2f}%")
    return missing_percentage

def check_feature_importance(df, target="Cena", feature="Rok_produkcji"):
    df_clean = df.dropna(subset=[target, feature]).copy()
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).fillna('missing')
        df_clean[col] = df_clean[col].astype('category').cat.codes
    
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance (top 10):")
    print(importance.head(10))
   
    r2_with = r2_score(y_test, model.predict(X_test))
    
    X_train_no_year = X_train.drop(columns=[feature])
    X_test_no_year = X_test.drop(columns=[feature])
    model_no_year = RandomForestRegressor(random_state=42)
    model_no_year.fit(X_train_no_year, y_train)
    r2_without = r2_score(y_test, model_no_year.predict(X_test_no_year))
    
    print(f"\nR² z {feature}: {r2_with:.4f}")
    print(f"R² bez {feature}: {r2_without:.4f}")
    print(f"Spadek R² po usunięciu {feature}: {(r2_with - r2_without):.4f}")


def check_correlations(df, target="Rok_produkcji", other_features=None):
   
    if other_features is None:
        other_features = ["Generacja_pojazdu", "Marka_pojazdu", "Przebieg_km", "Moc_KM"]
    
    df_clean = df.dropna(subset=[target]).copy()
    
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).fillna('missing')
        df_clean[col] = df_clean[col].astype('category').cat.codes
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    corr_features = [col for col in other_features if col in numeric_cols]
    if corr_features:
        correlations = df_clean[corr_features + [target]].corr()
        
        print(f"\nKorelacje z {target}:")
        print(correlations[target].sort_values(ascending=False))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title(f"Korelacje z {target}")
        plt.show()
    else:
        print(f"Brak wybranych cech numerycznych do analizy korelacji.")

def analyze_year_importance(df):
    # missing_percentage = check_missing_percentage(df, "Moc_KM")
    # print(missing_percentage)

    # missing_percentage_year = check_missing_percentage(df, "Rok_produkcji")
    # print(missing_percentage_year)
    
    check_feature_importance(df, target="Cena", feature="Rok_produkcji")
    
    # check_correlations(df, target="Rok_produkcji")

TRAIN_FILE_NAME = "data/sales_ads_train_org.csv"
TEST_FILE_NAME = "data/sales_ads_test.csv"
df = pd.read_csv(TRAIN_FILE_NAME)
analyze_year_importance(df)
import pandas as pd
import numpy as np
from datetime import datetime

# UTILS
def calculate_max_mileage_for_new(df):
    new_cars = df[(df["Stan"] == "New") & df["Przebieg_km"].notna()]
    if len(new_cars) == 0:
        return 10000  
    max_mileage = int(np.percentile(new_cars["Przebieg_km"], 95))
    print(max_mileage)
    return max(max_mileage, 1000)

def fill_production_year_from_generation(row):
    if pd.notna(row["Rok_produkcji"]):
        return row["Rok_produkcji"]
    if pd.isna(row["Generacja_pojazdu"]):
        return row["Rok_produkcji"]
    
    gen_str = str(row["Generacja_pojazdu"])
    if "(" not in gen_str or ")" not in gen_str:
        return row["Rok_produkcji"]
    
    years_part = gen_str[gen_str.find("(")+1:gen_str.find(")")]
    if "-" not in years_part:
        return row["Rok_produkcji"]
    
    years = years_part.split("-")
    min_year = int(years[0].strip())
    max_year = int(years[1].strip()) if years[1].strip() else 2021
    
    avg_year = (min_year + max_year) // 2
    return avg_year

def fill_production_year_from_mileage(df):
    ref_df = df[df["Rok_produkcji"].notna() & df["Przebieg_km"].notna()].copy()
    
    def fill_year(row):
        if pd.notna(row["Rok_produkcji"]):
            return row["Rok_produkcji"]
        if pd.isna(row["Przebieg_km"]):
            return row["Rok_produkcji"]
        
        mileage = row["Przebieg_km"]
        if mileage <= 20:
            return 2021
        
        category = row["Typ_nadwozia"]
        similar_cars = ref_df[ref_df["Typ_nadwozia"] == category]
        if len(similar_cars) == 0:
            return row["Rok_produkcji"]
        
        mileage_range = (mileage * 0.9, mileage * 1.1)
        candidates = similar_cars[(similar_cars["Przebieg_km"] >= mileage_range[0]) & (similar_cars["Przebieg_km"] <= mileage_range[1])]
        
        if len(candidates) == 0:
            return row["Rok_produkcji"]
        
        candidates = candidates.iloc[np.argsort(np.abs(candidates["Przebieg_km"] - mileage))].head(5)
        avg_year = int(candidates["Rok_produkcji"].mean())
        return avg_year
    
    df["Rok_produkcji"] = df.apply(fill_year, axis=1)
    return df

def remove_outliers(df, iqr_scale=1.5):
    numeric_cols = ["Przebieg_km", "Moc_KM", "Pojemnosc_cm3", "Wiek_pojazdu"]
    
    df["Premium_Factor"] = df["Moc_KM"] * df["Pojemnosc_cm3"] / 1000
    premium_threshold = np.percentile(df[df["Premium_Factor"].notna()]["Premium_Factor"], 95)
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_scale * IQR
            upper_bound = Q3 + iqr_scale * IQR
            df.loc[df["Premium_Factor"] >= premium_threshold, col] = df.loc[df["Premium_Factor"] >= premium_threshold, col]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound) | (df["Premium_Factor"] >= premium_threshold)]
    
    Q1 = df["Cena"].quantile(0.25)
    Q3 = df["Cena"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_scale * IQR
    upper_bound = Q3 + iqr_scale * IQR
    
    df.loc[df["Premium_Factor"] >= premium_threshold, "Cena"] = df.loc[df["Premium_Factor"] >= premium_threshold, "Cena"]
    df = df[(df["Cena"] >= lower_bound) & (df["Cena"] <= upper_bound) | 
            ((df["Premium_Factor"] >= premium_threshold) & (df["Cena"] <= upper_bound * 1.5))]  # Ochrona przed absurdalnymi cenami
    
    df = df.drop(columns=["Premium_Factor"])
    
    return df

# MAIN CODE
TRAIN_FILE_NAME = "data/sales_ads_train.csv"
df = pd.read_csv(TRAIN_FILE_NAME)

drop_columns = [
    "ID", "Lokalizacja_oferty", "Data_pierwszej_rejestracji", "Data_publikacji_oferty", "Wyposazenie", "Emisja_CO2", "Kolor"
]
df = df.drop(columns=drop_columns)

df = df[df["Waluta"].isin(["PLN", "EUR"])]

numeric_columns = ["Rok_produkcji", "Przebieg_km", "Moc_KM", "Pojemnosc_cm3", "Liczba_drzwi"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

df["Rok_produkcji"] = df.apply(fill_production_year_from_generation, axis=1)

df = fill_production_year_from_mileage(df)

max_mileage_new = calculate_max_mileage_for_new(df)

def fill_stan_column(row):
    if pd.isna(row["Stan"]):
        if pd.notna(row["Rok_produkcji"]) and pd.notna(row["Przebieg_km"]):
            if (row["Rok_produkcji"] >= 2018) and (row["Przebieg_km"] <= max_mileage_new):
                return "New"
            else:
                return "Used"
        return "Unknown"
    return row["Stan"]

df["Stan"] = df.apply(fill_stan_column, axis=1)

df["Stan"] = pd.Categorical(df["Stan"], categories=["New", "Used", "Unknown"], ordered=False)

df["Wiek_pojazdu"] = 2021 - df["Rok_produkcji"]

df = remove_outliers(df)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data/cleaned_sales_ads_train_{timestamp}.csv"

df.to_csv(filename, index=False)
print(f"Zapisano plik jako: {filename}")
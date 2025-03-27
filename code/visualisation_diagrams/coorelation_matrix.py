import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

TRAIN_FILE_NAME = "data/sales_ads_train.csv"

df = pd.read_csv(TRAIN_FILE_NAME)

df = df.dropna(subset=['Cena', 'Rok_produkcji', 'Przebieg_km', 'Moc_KM', 'Pojemnosc_cm3', 'Emisja_CO2', 'Liczba_drzwi'])

numeric_cols = ['Cena', 'Rok_produkcji', 'Przebieg_km', 'Moc_KM', 'Pojemnosc_cm3', 'Emisja_CO2', 'Liczba_drzwi']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

max_reasonable_price = 1000000
df = df[(df['Cena'] >= 1000) & (df['Cena'] <= max_reasonable_price)]

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')

plt.title('Mapa korelacji - interaktywny wykres z wartościami', fontsize=14)
plt.xlabel('Zmienne', fontsize=12)
plt.ylabel('Zmienne', fontsize=12)

plt.show()
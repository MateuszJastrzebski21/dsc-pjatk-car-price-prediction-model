import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

TRAIN_FILE_NAME = "data/sales_ads_train.csv"
df = pd.read_csv(TRAIN_FILE_NAME)

df = df.dropna(subset=['Cena'])
df['Cena'] = pd.to_numeric(df['Cena'], errors='coerce')

max_reasonable_price = 1_000_000  
df = df[(df['Cena'] >= 1_000) & (df['Cena'] <= max_reasonable_price)]

outliers = df[df['Cena'] > max_reasonable_price]
if not outliers.empty:
    print("Znaleziono nierealistyczne ceny (powyżej 1,000,000 PLN):")
    print(outliers[['Marka_pojazdu', 'Model_pojazdu', 'Rok_produkcji', 'Cena']])

n = len(df)
k = int(1 + np.log2(n))

log_bins = np.logspace(np.log10(df['Cena'].min()), np.log10(df['Cena'].max()), k)

plt.figure(figsize=(12, 6))
sns.histplot(df['Cena'], bins=log_bins, kde=True, color='skyblue')

plt.xscale('log')
plt.xlabel("Cena w PLN (log scale)")
plt.ylabel("Liczba samochodów")
plt.title("Rozkład ceny samochodów (skala logarytmiczna)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.xticks(log_bins, labels=[f"{int(x):,}" for x in log_bins], rotation=45)

plt.show()

print("\nPodstawowe statystyki cen (w PLN):")
print(df['Cena'].describe())

import pandas as pd
import matplotlib.pyplot as plt

import glob
import os

csv_files = glob.glob("data/origin/price_correlation_by_country_*.csv")
latest_file = max(csv_files, key=os.path.getctime)
df = pd.read_csv(latest_file)

audi_df = df[df["Marka"] == "Audi"]

top_5_audi = audi_df.nlargest(5, "Liczba_wystapien")

top_5_audi["Grupa"] = top_5_audi.apply(
    lambda row: f"{row['Marka']} {row['Model']} {row['Rok_produkcji']} {row['Rodzaj_paliwa']} poj. {row['Pojemnosc']} (liczność: {row['Liczba_wystapien']})",
    axis=1
)

plt.figure(figsize=(14, 6))
bars = plt.bar(top_5_audi["Grupa"], top_5_audi["Korelacja_Cena_Kraj"], color="skyblue", edgecolor="black")

plt.xlabel("Grupa pojazdów")
plt.ylabel("Korelacja między ceną a krajem pochodzenia")
plt.title("Korelacja ceny z krajem pochodzenia dla 5 najliczniejszych grup Audi")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha="center", va="bottom" if yval >= 0 else "top")

plt.tight_layout()

plt.show()
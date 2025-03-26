import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

TRAIN_FILE_NAME = "data/sales_ads_train.csv"
df = pd.read_csv(TRAIN_FILE_NAME)

features = [
    "Rok_produkcji", "Przebieg_km", "Moc_KM", "Pojemnosc_cm3", 
    "Rodzaj_paliwa", "Naped", "Skrzynia_biegow", "Typ_nadwozia"
]

df = df[features].dropna()  # Usuwamy brakujące wartości

categorical_features = ["Rodzaj_paliwa", "Naped", "Skrzynia_biegow", "Typ_nadwozia"]
df_encoded = df.copy()

for col in categorical_features:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])  # Zamiana na liczby

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

inertia = []
silhouette_scores = []
K_range = range(2, 15)

for k in K_range:
    print("rozpoczynam iteracje dla: ",k)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print("wykonano iteracje dla: ", k, " z wynikiem: ", kmeans.inertia_)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Liczba klastrów k")
plt.ylabel("Inertia")
plt.title("Metoda łokcia")

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', color='red')
plt.xlabel("Liczba klastrów k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")

plt.tight_layout()
plt.show()

optimal_k = 5 

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df.groupby("Cluster").mean()) 

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["Cluster"], palette="viridis", alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Wizualizacja klastrów (PCA)")
plt.legend(title="Cluster")
plt.show()

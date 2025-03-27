import pandas as pd

TRAIN_FILE_NAME = "data/sales_ads_train.csv"

df = pd.read_csv(TRAIN_FILE_NAME)

missing_values = df.isnull().sum()

missing_values_df = pd.DataFrame(missing_values, columns=['Liczba brakujących wartości'])

missing_values_df.to_csv('missing_values_report.csv')

print("Liczba brakujących wartości:")
print(missing_values_df)
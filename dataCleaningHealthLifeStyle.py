import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def IQRMode(columnName, df):
    Q1 = df[columnName].quantile(0.25)
    Q3 = df[columnName].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.0 * IQR
    upper_bound = Q3 + 1.0 * IQR

    outliers = df[(df[columnName] < lower_bound) | (df[columnName] > upper_bound)]
    print(f"\nOutlierek ({columnName}): {len(outliers)} db")
    if not outliers.empty:
        print(outliers[[columnName]])

    # negatív értékeket NaN-ra állítjuk
    df[columnName] = df[columnName].where(df[columnName] >= 0, np.nan)

    # csak az érvényes sorokat tartjuk meg
    df = df[(df[columnName] >= lower_bound) & (df[columnName] <= upper_bound)]

    return df

def showHistrograms(column):
    # Hisztogram
    plt.hist(df[column], bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')

    # Haranggörbe
    mean, std = df[column].mean(), df[column].std()
    x = np.linspace(df[column].min(), df[column].max(), 100)
    plt.plot(x, norm.pdf(x, mean, std), color='black', linewidth=2)

    plt.title(f"{column} eloszlása és normális eloszlás haranggörbéje")
    plt.xlabel(column)
    plt.ylabel("Értékek")
    plt.show()

#CSV beolvasása és kiíratása

df = pd.read_csv("health_lifestyle_dataset.csv")
print(df)

#id oszlop droppolása, nem szükséges a továbbiakban, illetve a felállított hipotéziseimhez az alábbiak sem: systolic_bp, diastolic_bp
df = df.drop(columns=['id', 'systolic_bp', 'diastolic_bp'])
print(df)

#megnézzük van e olyan oszlop, ami null értéket tartalmaz (nincs)
print(df.isnull().sum())

#outlierek detektálása, a numerikus oszlopokra (IQE teszttel egyik oszlopnál sincsen outlier)
for col in ["age", "bmi", "daily_steps", "sleep_hours", "water_intake_l",
            "calories_consumed", "resting_hr", "cholesterol"]:
    df = IQRMode(col, df)
    
#a numerikus oszlopok vizualizálása, hasonlít e haranggörbére
for col in ["age", "bmi", "daily_steps", "sleep_hours", "water_intake_l",
            "calories_consumed", "resting_hr", "cholesterol"]:
    #showHistrograms(col)
    print("itt lennenek a hisztogrammok")

#megnézzük vannak e irraális értékek az adathalmazban
for col in df.columns:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")
    
# ha lennének kiugró vagy nem normális értékek, akkor ezzel be lehetne állítani, hogy csak az adott tartományon belülieket tartsa meg
df = df[
    (df["age"].between(1, 90)) &
    (df["bmi"].between(9, 65)) &
    (df["daily_steps"].between(0, 20000)) &
    (df["sleep_hours"].between(2, 12)) &
    (df["water_intake_l"].between(0, 6)) &
    (df["calories_consumed"].between(1200, 5000)) &
    (df["resting_hr"].between(50, 110)) &
    (df["cholesterol"].between(140, 310))
]

# adattranszformáció - Female és Male változók 0,1-re való transzformálása
# Nem -> one-hot encoding (külön oszlop minden kategóriának)
ohe = OneHotEncoder(sparse_output=False, drop=None)  # drop=None: megtartjuk az összeset
ohe_cols = ohe.fit_transform(df[["gender"]])
ohe_colnames = [f"gender_{cat}" for cat in ohe.categories_[0]]
df[ohe_colnames] = ohe_cols
df = df.drop(columns = ['gender'])

# a hipotézisemhez szükséges az egyes értékeket összehasonlítani a többivel, pl bmi és calorie intake, ezért min-max scalelem a numerikus oszlopokat
# mivel nálam nincsenek extrém outlierek, és vizuélis szempontból is jobb a minmax, ezért alkalmazom azt

minmax = MinMaxScaler()
df["age_minmax"] = minmax.fit_transform(df[["age"]])
df["bmi_minmax"] = minmax.fit_transform(df[["bmi"]])
df["daily_steps_minmax"] = minmax.fit_transform(df[["daily_steps"]])
df["sleep_hours_minmax"] = minmax.fit_transform(df[["sleep_hours"]])
df["water_intake_l_minmax"] = minmax.fit_transform(df[["water_intake_l"]])
df["calories_consumed_minmax"] = minmax.fit_transform(df[["calories_consumed"]])
df["resting_hr_minmax"] = minmax.fit_transform(df[["resting_hr"]])
df["cholesterol_minmax"] = minmax.fit_transform(df[["cholesterol"]])

# eltávolítjuk az eredeti oszlopokat
df = df.drop(columns = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l', 'calories_consumed', 'resting_hr', 'cholesterol'])

# elmentjük a módosított csv
df.to_csv("health_lifestyle_modified.csv")

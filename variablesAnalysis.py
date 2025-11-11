import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, linregress
import numpy as np
import seaborn as sns

def showHistogram(column):

    # hisztogram megrajzolása
    plt.figure(figsize=(8, 5))
    plt.hist(healthDataSet[column], bins=40, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(f"{column} eloszlása mediánnal, átlaggal, kvartilisekkel és szórással")
    plt.xlabel(column)
    plt.ylabel("Gyakoriságok")

    # átlag
    plt.axvline(healthDataSet[column].mean(), color="blue", linestyle="-", linewidth=4, label=f"Átlag = {healthDataSet[column].mean():.2f}")
    # medián
    plt.axvline(healthDataSet[column].median(), color="black", linestyle="-.", linewidth=4, label=f"Medián = {healthDataSet[column].median():.2f}")
    # kvartilisek
    plt.axvline(healthDataSet[column].quantile(0.25), color="red", linestyle="dotted", linewidth=4, label=f"Q1 = {healthDataSet[column].quantile(0.25):.2f}")
    plt.axvline(healthDataSet[column].quantile(0.75), color="red", linestyle="dotted", linewidth=4, label=f"Q3 = {healthDataSet[column].quantile(0.75):.2f}")
    # szórás
    plt.axvline(healthDataSet[column].mean()-healthDataSet[column].std(), color="green", linestyle="solid", linewidth=2, label=f"Átlag-Szórás = {healthDataSet[column].mean()-healthDataSet[column].std():.2f}")
    plt.axvline(healthDataSet[column].mean()+healthDataSet[column].std(), color="green", linestyle="solid", linewidth=2, label=f"Átlag+Szórás = {healthDataSet[column].mean()+healthDataSet[column].std():.2f}")

    # jelmagyarázat kirajzolása
    plt.legend()
    plt.show()
    
def showDensityistogram(column):
    plt.figure(figsize=(8, 5))
    # density true -> nem a gyakoriságok láthatóak az y tengelyen
    plt.hist(healthDataSet[column], bins=40, color="skyblue", edgecolor="black", alpha=0.7, density=True)
    plt.title(f"{column} sűrűségfüggvénye")
    plt.xlabel(column)
    plt.ylabel("Gyakoriságok")
    
    kde = gaussian_kde(healthDataSet[column].dropna())
    x = np.linspace(healthDataSet[column].dropna().min(), healthDataSet[column].dropna().max(), 200)
    plt.plot(x, kde(x), color="yellow", linewidth=2, label="Sűrűségfüggvény")
    
def showBoxPlot(column):
    plt.figure(figsize=(6, 6))
    healthDataSet[column].plot(kind="box", vert=True)
    plt.title(f"{column} - boxplot")
    plt.ylabel(column)
    plt.show()
    plt.close()
    
#-------------------------------------------------------------------------------------------------------------#

healthDataSet = pd.read_csv("health_lifestyle_modified.csv")
healthDataSetOriginal = pd.read_csv("health_lifestyle_dataset.csv")

#-------------------------------------------------------------------------------------------------------------#

# leíró statisztikája az adathalmaznak
print(healthDataSet.describe())

#-------------------------------------------------------------------------------------------------------------#

# egyes oszlopok - numerikus változók - leíró statisztikája

columns = ['age_minmax', "bmi_minmax", "daily_steps_minmax", "sleep_hours_minmax", 
           "water_intake_l_minmax", "calories_consumed_minmax", "resting_hr_minmax", "cholesterol_minmax"]

originalColumns = ['age', "bmi", "daily_steps", "sleep_hours", 
           "water_intake_l", "calories_consumed", "resting_hr", "cholesterol"]

for currentColumn in columns:
    print(healthDataSet[currentColumn])
    print(f"Terjedelem: {healthDataSet[currentColumn].max()-healthDataSet[currentColumn].min()}")
    showHistogram(currentColumn)
    showBoxPlot(currentColumn)
    showDensityistogram(currentColumn)
    
# kategorikus változók statisztikája

categoryColumns = ['smoker', 'alcohol', 'family_history', 'disease_risk', 'gender_Female']

for currentCategoryColumn in categoryColumns:
    print(healthDataSet[currentCategoryColumn].describe())
    healthDataSet[currentCategoryColumn].value_counts().plot(kind='bar')
    plt.title(f"{currentCategoryColumn} eloszlása")
    plt.ylabel("Darabszám")
    plt.show()
    
#-------------------------------------------------------------------------------------------------------------#

# kétváltozós elemzés a resting_hr_minmax és bmi_minmax között

print(healthDataSet[['resting_hr_minmax', 'bmi_minmax']].describe())
corr = healthDataSet['resting_hr_minmax'].corr(healthDataSet['bmi_minmax'])
print(f"Korreláció az resting_hr és bmi között: {corr:.2f}")

plt.figure(figsize=(7,5))
plt.scatter(healthDataSet['resting_hr_minmax'], healthDataSet['bmi_minmax'], alpha=0.6, color='blue', edgecolor='k')
plt.title("Kapcsolat a nyugalmi pulzus és a bmi között")
plt.xlabel("Resting hr (skálázott)")
plt.ylabel("Bmi (skálázott)")
plt.grid(alpha=0.4)
plt.show()

x = healthDataSet['water_intake_l_minmax']
y = healthDataSet['bmi_minmax']

# van minimális lineáris korreláció, így meghatározzuk a regressziós egyenest us
result = linregress(x, y)
print("R =", result.rvalue, ", p-value =", result.pvalue)
print("Slope: ", result.slope, result.stderr)
print("Intercept: ", result.intercept, result.intercept_stderr)

plt.plot(x, y, 'o', label='original data')
plt.plot(x, result.slope * np.array(x) + result.intercept, 'r', label='fitted line')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------------------------#

# korrelációs mátrix minden oszlop között

corr_matrix = healthDataSet.corr(method="pearson")
print(corr_matrix)

# hőtérképes megjelenítés
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Korrelációs mátrix")
plt.show()

#-------------------------------------------------------------------------------------------------------------#




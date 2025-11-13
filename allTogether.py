import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import gaussian_kde, linregress, kstest, kruskal
import seaborn as sns
from sklearn.isotonic import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, silhouette_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

print("---------------------- 1. Adattisztítás ----------------------")

#---------------------------------------------------------------------------------#

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
    # hisztogram
    plt.hist(df[column], bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')

    # haranggörbe
    mean, std = df[column].mean(), df[column].std()
    x = np.linspace(df[column].min(), df[column].max(), 100)
    plt.plot(x, norm.pdf(x, mean, std), color='black', linewidth=2)

    plt.title(f"{column} eloszlása és normális eloszlás haranggörbéje")
    plt.xlabel(column)
    plt.ylabel("Értékek")
    plt.show()
    
#---------------------------------------------------------------------------------#

#CSV beolvasása és kiíratása

df = pd.read_csv("health_lifestyle_dataset.csv")

#id oszlop droppolása, nem szükséges a továbbiakban, illetve a felállított hipotéziseimhez az alábbiak sem: systolic_bp, diastolic_bp
df = df.drop(columns=['id', 'systolic_bp', 'diastolic_bp'])
print(df)

#megnézzük van e olyan oszlop, ami null értéket tartalmaz (nincs)
print(df.isnull().sum())

#outlierek detektálása, a numerikus oszlopokra (IQR teszttel egyik oszlopnál sincsen outlier)
for col in ["age", "bmi", "daily_steps", "sleep_hours", "water_intake_l",
            "calories_consumed", "resting_hr", "cholesterol"]:
    df = IQRMode(col, df)
    
#a numerikus oszlopok vizualizálása, hasonlít e haranggörbére
for col in ["age", "bmi", "daily_steps", "sleep_hours", "water_intake_l",
            "calories_consumed", "resting_hr", "cholesterol"]:
    showHistrograms(col)
    print("itt lennenek a hisztogrammok")

#---------------------------------------------------------------------------------#

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

#---------------------------------------------------------------------------------#

# adattranszformáció - Female és Male változók 0,1-re való transzformálása
# Nem -> one-hot encoding (külön oszlop minden kategóriának)
ohe = OneHotEncoder(sparse_output=False, drop=None)  # drop=None: megtartjuk az összeset
ohe_cols = ohe.fit_transform(df[["gender"]])
ohe_colnames = [f"gender_{cat}" for cat in ohe.categories_[0]]
df[ohe_colnames] = ohe_cols
df = df.drop(columns = ['gender'])

#---------------------------------------------------------------------------------#

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

#---------------------------------------------------------------------------------#

# eltávolítjuk az eredeti oszlopokat
df = df.drop(columns = ['age', 'bmi', 'daily_steps', 'sleep_hours', 'water_intake_l', 'calories_consumed', 'resting_hr', 'cholesterol'])

#---------------------------------------------------------------------------------#

# elmentjük a módosított csv
df.to_csv("health_lifestyle_modified.csv")

#---------------------------------------------------------------------------------#

print("---------------------- 2. Változók elemzése ----------------------")

#---------------------------------------------------------------------------------#

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
    
#-------------------------------------------------------------------------------------------------------------#
    
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

# van minimális lineáris korreláció, így meghatározzuk a regressziós egyenest is
result = linregress(x, y)
print("R: ", result.rvalue, ", p-value =", result.pvalue)
print("Slope: ", result.slope, result.stderr)
print("Intercept: ", result.intercept, result.intercept_stderr)

#egyébként a p-value-ból láthatjuk, hogy nem szignifikáns a kapcsolat

plt.plot(x, y, 'o', label='adathalmaz')
plt.plot(x, result.slope * np.array(x) + result.intercept, 'r', label='Egy nap alatt megivott víz és a bmi kapcsolata')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------------------------#

# korrelációs mátrix minden oszlop között

corr_matrix = healthDataSet.corr(method="pearson")
print(corr_matrix)

# hőtérképes megjelenítés
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Korrelációs mátrix")
plt.show()

#-------------------------------------------------------------------------------------------------------------#

print("---------------------- 3. Hipotézisek és elemzésük ----------------------")

#-------------------------------------------------------------------------------------------------------------#

def checkIfVariablesAreNormallyDistributed(list1, name1):
    #Normál eloszlás teszt
    #H0: a minta normál eloszlású
    z = (list1 - np.mean(list1)) / np.std(list1)
    statistic, p_value = stats.kstest(z, 'norm')
    print(f"Normál eloszlás teszt: {name1}", statistic, p_value)
    #ha p < 0.05 -> elvetjük H0-at

def calculateCorrelation(column1, column2):
    corr, p_value = spearmanr(healthDataframe[column1], healthDataframe[column2])
    print('Spearman korreláció: %.3f' % corr, ', p-value =', p_value)
    if(p_value > 0.05):
        print("A korreláció érték korrelációt nem mutat, H0-t elvetjük.\n")
    else:
        print("A korreláció érték korrelációt mutat, H0-t megtartjuk.\n")
    
def calculateMannWhitney(firstList, secondList, alternativeType, hypothesisNumber):
    statistic, p_value = stats.mannwhitneyu(firstList, secondList, alternative=alternativeType)
    print(f"Statistics for {hypothesisNumber}: {statistic} p value for {hypothesisNumber}: {p_value}")
    
def calculateLinearRegression(column1, column2):
    x = healthDataframe[column1]
    y = healthDataframe[column2]
    result = linregress(x, y)
    print(f"R {column1} & {column2}= ", result.rvalue, ", p-value =", result.pvalue)
    print(f"Slope {column1} & {column2}: ", result.slope, result.stderr)
    print(f"Intercept {column1} & {column2}: {result.intercept, result.intercept_stderr}\n")
    
    return [x, y, result]

#-----------------------------------------------------------------------------------------#

healthDataframe = pd.read_csv("health_lifestyle_modified.csv")
#print(healthDataframe)

#-----------------------------------------------------------------------------------------#

print("1. A nagyobb BMI értékkel rendelkezők több kalóriát fogyasztanak el egy nap.")
# H0: a két minta lineáris korrelációban van egymással
checkIfVariablesAreNormallyDistributed(healthDataframe['bmi_minmax'], 'BMI')
checkIfVariablesAreNormallyDistributed(healthDataframe['calories_consumed_minmax'], 'Calories consumed')

calculateCorrelation('bmi_minmax', 'calories_consumed_minmax')
#p_value > 0.05 -> H0-at elvetjük

print("#-----------------------------------------------------------------------------------------#")

print("2. Aki több vizet iszik egy nap, többet is alszik.")
# H0: a két minta lineáris korrelációban van egymással
checkIfVariablesAreNormallyDistributed(healthDataframe['water_intake_l_minmax'], 'Water intake')
checkIfVariablesAreNormallyDistributed(healthDataframe['sleep_hours_minmax'], 'Sleep hours')
calculateCorrelation('water_intake_l_minmax', 'sleep_hours_minmax')
#p_value > 0.05 -> H0-at elvetjük

print("#-----------------------------------------------------------------------------------------#")

print("3. A dohányzók körében magasabb a nyugalmi szívverés, mint a nem dohányzók körében.")
notSmokersHR = []
smokersHR = []
index = 0
for i in healthDataframe['smoker']:
    if i == 0:
        notSmokersHR.append(healthDataframe.loc[index, 'resting_hr_minmax'])
    else:
        smokersHR.append(healthDataframe.loc[index, 'resting_hr_minmax'])
    index += 1
    
checkIfVariablesAreNormallyDistributed(smokersHR, 'Smokers heartrate')
checkIfVariablesAreNormallyDistributed(notSmokersHR, 'Not smokers heartrate')

# Mann–Whitney U próba
calculateMannWhitney(smokersHR, notSmokersHR, 'greater', 'H3')
print("H0-at elvetjük: tehát a dohányzók körében nem feltétlen magasabb a nyugalmi szívverés\n")
#p > 0.05 -> H0-at elvetjük: tehát a dohányzók körében nem feltétlen magasabb a nyugalmi szívverés

print("#-----------------------------------------------------------------------------------------#")

print("4. A több napi lépésszámot megtevők, több kalóriát fogyasztanak el, és többet alszanak.\n")
#lineáris regresszió vizsgálata
results = calculateLinearRegression('daily_steps_minmax', 'sleep_hours_minmax')
results2 = calculateLinearRegression('daily_steps_minmax', 'calories_consumed_minmax')

plt.plot(results[0], results[1], 'o', color='black', label='Alapadatok')
plt.plot(results[0], results[2].slope * np.array(results[0]) + results[2].intercept, 'r', label='Lépésszám és alvás közötti kapcsolat')

plt.plot(results2[0], results2[2].slope * np.array(results2[0]) + results2[2].intercept, 'r', color = 'blue', label='Lépésszám és kalóriabevitel közötti kapcsolat')

plt.legend()
plt.show()

print("#-----------------------------------------------------------------------------------------#")

print("\n5. A fiatal, középkorú és idős csoportok átlaga nem különbözik a napi vízbevitelben.")
# felosztjuk 3 csoportra age szerint az adatokat

healthDataframe['age_group'] = pd.qcut(
    healthDataframe['age_minmax'],
    q=3,
    labels=['young', 'middle-aged', 'old']
)

water_intake_youngs = healthDataframe.loc[healthDataframe['age_group'] == 'young', 'water_intake_l_minmax']
water_intake_middles = healthDataframe.loc[healthDataframe['age_group'] == 'middle-aged', 'water_intake_l_minmax']
water_intake_olds = healthDataframe.loc[healthDataframe['age_group'] == 'old', 'water_intake_l_minmax']

checkIfVariablesAreNormallyDistributed(water_intake_youngs, 'Water intake youngsters')
checkIfVariablesAreNormallyDistributed(water_intake_middles, 'Water intake middle aged')
checkIfVariablesAreNormallyDistributed(water_intake_olds, 'Water intake oldies')

# Kruskal–Wallis teszt
stat, p = kruskal(water_intake_youngs, water_intake_middles, water_intake_olds)

print(f"Kruskal-Wallis teszt statisztikája: {stat:.3f}")
print(f"p-értéke: {p:.4f}")
print("Szignifikáns különbség nincs a csoportok között így a H5 hipotézisem teljesül.")

#vizualizáció
plt.figure(figsize=(6,5))
plt.boxplot([water_intake_youngs, water_intake_middles, water_intake_olds], tick_labels=['fiatal', 'középkorú', 'idős'])
plt.ylabel('Vízbevitel naponta')
plt.title('Vízbevitel csoportonként')
plt.show()

print("#-----------------------------------------------------------------------------------------#")

print("6. A férfiak esetében nagyobb arányban volt már a családban betegség, mint a nők körében.")

womenFamilyDisease = healthDataframe.loc[healthDataframe['gender_Female'] == 1, 'family_history'].tolist()
menFamilyDisease = healthDataframe.loc[healthDataframe['gender_Male'] == 1, 'family_history'].tolist()

checkIfVariablesAreNormallyDistributed(womenFamilyDisease, 'Woman family disease')
checkIfVariablesAreNormallyDistributed(menFamilyDisease, 'Men family disease')

# Mann–Whitney U próba
calculateMannWhitney(womenFamilyDisease, menFamilyDisease, 'less', 'H6')
print("H0-at megtartjuk: tehát a férfiak esetébem nagyobb arányban volt már a családban betegség\n")
#p < 0.05 -> H0-at megtartjuk: tehát a férfiak esetébem nagyobb arányban volt már a családban betegség

print("#-----------------------------------------------------------------------------------------#")

print("7. A fiatalabbak több lépést tesznek egy nap, mint az idősebbek.")
#először el kell döntenünk, hogy honnantól számítjuk a fiatalokat fiataloknak, időseket időseknek
ageAverage = healthDataframe['age_minmax'].mean()
ageStandardDeviation = healthDataframe['age_minmax'].std()

youngsters = healthDataframe.loc[healthDataframe['age_minmax'] < (ageAverage - (ageStandardDeviation / 2)), 'daily_steps_minmax']
oldies = healthDataframe.loc[healthDataframe['age_minmax'] > (ageAverage - (ageStandardDeviation / 2)), 'daily_steps_minmax']

checkIfVariablesAreNormallyDistributed(oldies, 'Olders daily steps')
checkIfVariablesAreNormallyDistributed(youngsters, 'Youngers daily steps')

# Mann–Whitney U próba
calculateMannWhitney(oldies, youngsters, 'less', 'H7')
print("H0-at elvetjük: tehát a fiatalabbak nem feltétlen tesznek meg több lépést, mint az idősek.\n")
#p > 0.05 -> H0-at elvetjük: tehát a fiatalabbak nem feltétlen tesznek meg több lépést, mint az idősek.

print("#-----------------------------------------------------------------------------------------#")

print("8. A nők koleszterin szintje alacsonyabb, mint a férfiaké.")
womenChLevel = healthDataframe.loc[healthDataframe['gender_Female'] == 1, 'cholesterol_minmax']
menChLevel = healthDataframe.loc[healthDataframe['gender_Male'] == 1, 'cholesterol_minmax']

checkIfVariablesAreNormallyDistributed(womenChLevel, 'Woman ch level')
checkIfVariablesAreNormallyDistributed(menChLevel, 'Men ch level')

# Mann–Whitney U próba
calculateMannWhitney(womenChLevel, menChLevel, 'less', 'H8')
print("H0-at elvetjük: tehát a nők koleszterin szintje nem feltétlen alacsonyabb, mint a férfiaké\n")
#p > 0.05 -> H0-at elvetjük: tehát a nők koleszterin szintje nem feltétlen alacsonyabb, mint a férfiaké

print("#-----------------------------------------------------------------------------------------#")

print("9. Minél több kalóriát visz be, annál kevesebbet iszik.")
# H0: a két minta lineáris korrelációban van egymással
checkIfVariablesAreNormallyDistributed(healthDataframe['calories_consumed_minmax'], 'Calories consumed')
checkIfVariablesAreNormallyDistributed(healthDataframe['water_intake_l_minmax'], 'Water intake')

calculateCorrelation('calories_consumed_minmax', 'water_intake_l_minmax')
#print("A korreláció érték gyenge, negatív korrelációt mutat, H0-t megtartjuk.\n")
#p_value < 0.05 -> H0-at megtartjuk

result = calculateLinearRegression('calories_consumed_minmax', 'water_intake_l_minmax')

plt.plot(result[0], result[1], 'o', label='Alapadatok')
plt.plot(result[0], result[2].slope * np.array(result[0]) + result[2].intercept, 'r', label='Kalória és vízbevitel közötti kapcsolat')
plt.legend()
plt.show()

#---------------------------------------------------------------------------------#

print("---------------------- 4. Klaszterezés ----------------------")

#---------------------------------------------------------------------------------#

# adatok betöltése és oszlopok kiválasztása, amelyeket szeretnénk klaszterezni
healthDataSet = pd.read_csv("health_lifestyle_modified.csv")
features = ['bmi_minmax', 'calories_consumed_minmax', 'daily_steps_minmax']
X = healthDataSet[features].values

#---------------------------------------------------------------------------------#

# normalizálás
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#---------------------------------------------------------------------------------#

# Könyök módszer: inertia értékek számítása 2–7 klaszterre
inertias = []
K = range(2, 7)
for k in K:
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Ábra
plt.figure(figsize=(7, 5))
plt.plot(K, inertias, 'o-', markersize=8)
plt.title("Könyök módszer az adathalmazra")
plt.xlabel("Klaszterek száma (k)")
plt.ylabel("Inertia (összesített hibatag)")
plt.xticks(K)
plt.grid(True)
plt.show()

#---------------------------------------------------------------------------------#

# könyökmódszer alapján
optimal_k = 4   
print(f"\n-------K-means klaszterezés k={optimal_k}-------")
kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_
print(f"Silhouette score for k={optimal_k}:", silhouette_score(X_scaled, labels, metric="euclidean"))

#---------------------------------------------------------------------------------#
# Klaszerek ábrázolása 

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10,7))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.7, s=3)
plt.scatter(centers_pca[:,0], centers_pca[:,1], c='black', s=200, marker='X', label='Centroidok')
plt.title("K means klaszterek (PCA 2D)")
plt.colorbar(scatter, label='Klaszter ID')
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

#---------------------------------------------------------------------------------#

cluster_means = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("\nK-means középpontok (eredeti 3D térben):")
print(cluster_means.round(3))

#---------------------------------------------------------------------------------#

print("---------------------- 5. Osztályozás ----------------------")

#---------------------------------------------------------------------------------#

#az előzőekben létrehozott klaszterek
cluster_labels = kmeans.labels_
cluster_datas = X

cluster_names = {
    0: "Normál életmód",
    1: "Túlsúlyos",
    2: "Vékony, diétázó",
    3: "Vékony, gyors anyagcseréjű"
}
#klaszter nevek listává
target_names = [cluster_names[i] for i in sorted(cluster_names.keys())]

#70–15–15% felosztás
cluster_datas_train, cluster_data_temp, cluster_labels_train, cluster_labels_temp = train_test_split(
    cluster_datas, cluster_labels, test_size=0.30, stratify=cluster_labels, random_state=42
)

#a 30%-ot felezzük: 15% teszt és 15% validáció
cluster_datas_test, cluster_datas_validation, cluster_labels_test, cluster_labels_validation = train_test_split(
    cluster_data_temp, cluster_labels_temp, test_size=0.50, stratify=cluster_labels_temp, random_state=42
)

#kiírjuk a méreteket, hogy ellenőrizzük
print(f"Train méret: {len(cluster_labels_train)}  ({len(cluster_labels_train)/len(cluster_labels):.0%})")
print(f"Test  méret: {len(cluster_labels_test)}   ({len(cluster_labels_test)/len(cluster_labels):.0%})")
print(f"Valid méret: {len(cluster_labels_validation)}   ({len(cluster_labels_validation)/len(cluster_labels):.0%})")

#osztályozó modell felépítése a train halmazon
dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(cluster_datas_train, cluster_labels_train)

#előzetes értékelés a teszt halmazon
cluster_labels_pred_test_base = dt_base.predict(cluster_datas_test)
print("\n---------Baseline döntési fa — TESZT halmaz---------")
print("Accuracy:", accuracy_score(cluster_labels_test, cluster_labels_pred_test_base))
print(classification_report(cluster_labels_test, cluster_labels_pred_test_base, target_names=target_names))

#5-fold cross validation a TRAIN halmazon + hiperparaméter-hangolás
param_grid = {
    "criterion": ["gini", "entropy"],  # a legjobb elágazás eldöntése
    "max_depth": [None, 2, 3, 4, 5, 6], # a fa maximális mélysége
    "min_samples_leaf": [1, 2, 3, 5], # minimum elemszám egy levélen
    "ccp_alpha": [0.0, 0.001, 0.01], # metszési paraméter, a fa túl mély ágait visszavágja
    # cost-complexity pruning (költség–komplexitás metszés)
}

#a legjobb paraméterek megkeresése a decisionTree-hez
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1 # az összes elérhető processzormagot felhasználja a számításokhoz
)

grid.fit(cluster_datas_train, cluster_labels_train)
print("\n--------5-fold CV eredmények (TRAIN)-------------")
print("Legjobb paraméterek:", grid.best_params_)
print("CV átlag accuracy (best):", grid.best_score_)

# végső modell felépítése train+test (85%) adatokon a megtalált paraméterekkel
best_dt = DecisionTreeClassifier(**grid.best_params_, random_state=42) #a ** a szótár elemeit szétszedi kulcs=érték formában
cluster_datas_train_plus_test = np.vstack([cluster_datas_train, cluster_datas_test])
cluster_labels_train_plus_test = np.hstack([cluster_labels_train, cluster_labels_test])
best_dt.fit(cluster_datas_train_plus_test, cluster_labels_train_plus_test)

#végső validáció a 15%-on
cluster_labels_pred_val = best_dt.predict(cluster_datas_validation)
final_acc = accuracy_score(cluster_labels_validation, cluster_labels_pred_val)

print("\n--------Végső validáció — VALIDÁCIÓS halmaz--------")
print("Végső pontosság (accuracy):", final_acc)
print(classification_report(cluster_labels_validation, cluster_labels_pred_val, target_names=target_names))

#konfúziós mátrix a végső kiértékeléshez
cm = confusion_matrix(cluster_labels_validation, cluster_labels_pred_val)
disp = ConfusionMatrixDisplay(cm, display_labels=cluster_names)
fig, ax = plt.subplots(figsize=(4.5, 4))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Confusion matrix — Final validation")
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------#

print("---------------------- 6. Új adatok osztályozása ----------------------")

#beolvassuk az új adatokat
new_data = pd.read_csv("classification_new_data.csv")

#minmaxscaler
minmax = MinMaxScaler()
new_data["bmi_minmax"] = minmax.fit_transform(new_data[["bmi"]])
new_data["daily_steps_minmax"] = minmax.fit_transform(new_data[["daily_steps"]])
new_data["calories_consumed_minmax"] = minmax.fit_transform(new_data[["calories_consumed"]])

# eltávolítjuk az eredeti oszlopokat
new_data = new_data.drop(columns = [ 'bmi', 'daily_steps', 'calories_consumed'])

#csak azok az oszlopok, amiket az eredeti modell is használ
new_features = ['bmi_minmax', 'daily_steps_minmax', 'calories_consumed_minmax']
X_new = new_data[new_features].values

#ugyanaz a scaler, mint a modellhez
X_new_scaled = scaler.transform(X_new)

#osztályozás a legjobb paraméterekkel rendelkező fával
new_predictions = best_dt.predict(X_new_scaled)

#eredmények hozzáadása dataframehez
new_data["Predicted_Cluster"] = new_predictions
new_data["Cluster names"] = new_data["Predicted_Cluster"].map(cluster_names)

#néhány eredmény kiíratása
print("\nAz új adatok klaszterbecslései:")
print(new_data.head())

#eredmények mentése
new_data.to_csv("new_classification_result.csv", index=False)

#---------------------------------------------------------------------------------#


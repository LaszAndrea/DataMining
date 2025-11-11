import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, silhouette_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

#---------------------------------------------------------------------------------#

# adatok betöltése és oszlopok kiválasztása, amelyeket szeretnénk klaszterezni
healthDataSet = pd.read_csv("health_lifestyle_modified.csv")
features = ['bmi_minmax', 'calories_consumed_minmax', 'daily_steps_minmax']
X = healthDataSet[features].values

#---------------------------------------------------------------------------------#

# Könyök módszer: inertia értékek számítása 1–10 klaszterre
inertias = []
K = range(2, 7)
for k in K:
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(X)
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

# normalizálás
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#---------------------------------------------------------------------------------#

optimal_k = 4   # könyökmódszer alapján
print(f"\n=== K-means klaszterezés k={optimal_k} ===")
kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_
print(f"Silhouette score for k={optimal_k}:", silhouette_score(X, labels, metric="euclidean"))

#---------------------------------------------------------------------------------#

# Dimenziócsökkentés vizualizációhoz (PCA: 3D -> 2D)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

#---------------------------------------------------------------------------------#
# Klaszerek ábrázolása 

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10,7))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.7, s=3)
plt.title("K means klaszterek (PCA 2D)")
plt.colorbar(scatter, label='Klaszter ID')
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()

#---------------------------------------------------------------------------------#

cluster_means = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("\nK-means középpontok (eredeti 3D térben):")
print(cluster_means.round(3))

#---------------------------------------------------------------------------------#

y = kmeans.labels_

# 1) 70–15–15% felosztás
# Először train (70%) + ideiglenes (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
# Az ideiglenes 30%-ot felezzük: 15% teszt + 15% validáció
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train méret: {len(y_train)}  ({len(y_train)/len(y):.0%})")
print(f"Test  méret: {len(y_test)}   ({len(y_test)/len(y):.0%})")
print(f"Valid méret: {len(y_val)}   ({len(y_val)/len(y):.0%})")

# 2) Osztályozó modell felépítése (baseline) csak a TRAIN halmazon
dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train, y_train)

# 3) Ellenőrzés a TESZT adatokon (előzetes értékelés)
y_pred_test_base = dt_base.predict(X_test)
print("\n=== Baseline döntési fa — TESZT halmaz ===")
print("Accuracy:", accuracy_score(y_test, y_pred_test_base))
print(classification_report(y_test, y_pred_test_base, target_names=[f"Cluster {i}" for i in np.unique(y)]))

# 4) 5-fold cross validation a TRAIN halmazon + hiperparaméter-hangolás
param_grid = {
    "criterion": ["gini", "entropy"],  # a legjobb elágazás eldöntése
    "max_depth": [None, 2, 3, 4, 5, 6], # a fa maximális mélysége
    "min_samples_leaf": [1, 2, 3, 5], # minimum elemszám egy levélen
    "ccp_alpha": [0.0, 0.001, 0.01], # metszési paraméter, a fa túl mély ágait visszavágja
    # cost-complexity pruning (költség–komplexitás metszés)
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1 # az összes elérhető processzormagot felhasználja a számításokhoz
)
grid.fit(X_train, y_train)
print("\n=== 5-fold CV eredmények (TRAIN) ===")
print("Legjobb paraméterek:", grid.best_params_)
print("CV átlag accuracy (best):", grid.best_score_)

# 5) Végső modell felépítése TRAIN+TEST (85%) adatokon a megtalált hiperparaméterekkel
best_dt = DecisionTreeClassifier(**grid.best_params_, random_state=42)
# A ** a szótár elemeit szétszedi kulcs=érték formában
X_train_plus_test = np.vstack([X_train, X_test])
y_train_plus_test = np.hstack([y_train, y_test])
best_dt.fit(X_train_plus_test, y_train_plus_test)

# 6) Végső VALIDÁCIÓ (15% különálló, csak kiértékelésre)
y_pred_val = best_dt.predict(X_val)
final_acc = accuracy_score(y_val, y_pred_val)

print("\n=== Végső validáció — VALIDÁCIÓS halmaz ===")
print("Végső pontosság (accuracy):", final_acc)
print(classification_report(y_val, y_pred_val, target_names=[f"Cluster {i}" for i in np.unique(y)]))

# Konfúziós mátrix a végső értékeléshez
cm = confusion_matrix(y_val, y_pred_val)
disp = ConfusionMatrixDisplay(cm, display_labels=[f"Cluster {i}" for i in np.unique(y)])
fig, ax = plt.subplots(figsize=(4.5, 4))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Confusion matrix — Final validation")
plt.tight_layout()
plt.show()

# A végső fa vizualizálása
plt.figure(figsize=(14, 8))
plot_tree(
    best_dt,
    feature_names=features,
    class_names=[f"Cluster {i}" for i in np.unique(y)],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree — final model (trained on 85%)")
plt.tight_layout()
plt.show()


clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)

#---------------------------------------------------------------------------------#
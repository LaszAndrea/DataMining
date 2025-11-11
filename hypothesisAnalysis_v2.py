import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, kstest
from scipy.stats import kruskal
import scipy.stats as stats
from sklearn.isotonic import spearmanr

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

print("#-----------------------------------------------------------------------------------------#")

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
print(f"p-érték: {p:.4f}")
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

print("#-----------------------------------------------------------------------------------------#")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

healthDataframe = pd.read_csv("health_lifestyle_modified.csv")
#print(healthDataframe)

#-----------------------------------------------------------------------------------------#

print("1. A nagyobb BMI értékkel rendelkezők több kalóriát fogyasztanak el egy nap.")
# H0: a két minta lineáris korrelációban van egymással
corr, p_value = pearsonr(healthDataframe['bmi_minmax'], healthDataframe['calories_consumed_minmax'])
print('Pearsons korreláció: %.3f' % corr, ', p-value =', p_value)
print("A korreláció érték korrelációt nem mutat, H0-t elvetjük.\n")
#p_value > 0.05 -> H0-at elvetjük

#-----------------------------------------------------------------------------------------#

print("2. Aki több vizet iszik egy nap, többet is alszik.")
# H0: a két minta lineáris korrelációban van egymással
corr, p_value = pearsonr(healthDataframe['water_intake_l_minmax'], healthDataframe['sleep_hours_minmax'])
print('Pearsons korreláció: %.3f' % corr, ', p-value =', p_value)
print("A korreláció érték korrelációt nem mutat, H0-t elvetjük.\n")
#p_value > 0.05 -> H0-at elvetjük

#-----------------------------------------------------------------------------------------#

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
         
# Mann–Whitney U próba
statistic, p_value = stats.mannwhitneyu(smokersHR, notSmokersHR, alternative='greater')
print(f"Statistics for H3: {statistic} p value for H3: {p_value}")
print("H0-at elvetjük: tehát a dohányzók körében nem feltétlen magasabb a nyugalmi szívverés\n")
#p > 0.05 -> H0-at elvetjük: tehát a dohányzók körében nem feltétlen magasabb a nyugalmi szívverés

#-----------------------------------------------------------------------------------------#

print("4. A több napi lépésszámot megtevők, több kalóriát fogyasztanak el, és többet alszanak.\n")
#lineáris regresszió vizsgálata

x = healthDataframe['daily_steps_minmax']
y1 = healthDataframe['sleep_hours_minmax']
result = linregress(x, y1)
print("R daily steps & sleep hours= ", result.rvalue, ", p-value =", result.pvalue)
print("Slope daily steps & sleep hours: ", result.slope, result.stderr)
print(f"Intercept daily steps & sleep hours: {result.intercept, result.intercept_stderr}\n")

y2 = healthDataframe['calories_consumed_minmax']
result2 = linregress(x, y2)
print("R daily steps & calories consumed =", result2.rvalue, ", p-value =", result2.pvalue)
print("Slope daily steps & calories consumed: ", result2.slope, result2.stderr)
print("Intercept daily steps & calories consumed: ", result2.intercept, result2.intercept_stderr)

plt.plot(x, y1, 'o', color='black', label='Alapadatok')
plt.plot(x, result.slope * np.array(x) + result.intercept, 'r', label='Lépésszám és alvás közötti kapcsolat')

plt.plot(x, result2.slope * np.array(x) + result2.intercept, 'r', color = 'blue', label='Lépésszám és kalóriabevitel közötti kapcsolat')

plt.legend()
plt.show()

#-----------------------------------------------------------------------------------------#

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

#one-way anova
f_stat, p_value = stats.f_oneway(water_intake_youngs, water_intake_middles, water_intake_olds)
print(f"\nF-statisztika: {f_stat:.3f}")
print(f"p-érték: {p_value:.4f}\n")

#Tukey HSD vizsgálat
tukey = pairwise_tukeyhsd(endog=healthDataframe['water_intake_l_minmax'],
                          groups=healthDataframe['age_group'],
                          alpha=0.05)
print(tukey)

#vizualizáció
plt.figure(figsize=(6,5))
plt.boxplot([water_intake_youngs, water_intake_middles, water_intake_olds], labels=['fiatal', 'középkorú', 'idős'])
plt.ylabel('Vízbevitel naponta')
plt.title('Vízbevitel csoportonként')
plt.show()

#-----------------------------------------------------------------------------------------#

print("6. A férfiak esetében nagyobb arányban volt már a családban betegség, mint a nők körében.")

womenFamilyDisease = healthDataframe.loc[healthDataframe['gender_Female'] == 1, 'family_history'].tolist()
menFamilyDisease = healthDataframe.loc[healthDataframe['gender_Male'] == 1, 'family_history'].tolist()

# Mann–Whitney U próba
statistic, p_value = stats.mannwhitneyu(womenFamilyDisease, menFamilyDisease, alternative='less')
print(f"Statistics for H6: {statistic} p value for H6: {p_value}")
print("H0-at megtartjuk: tehát a férfiak esetébem nagyobb arányban volt már a családban betegség\n")
#p < 0.05 -> H0-at megtartjuk: tehát a férfiak esetébem nagyobb arányban volt már a családban betegség

#-----------------------------------------------------------------------------------------#

print("7. A fiatalabbak több lépést tesznek egy nap, mint az idősebbek.")
#először el kell döntenünk, hogy honnantól számítjuk a fiatalokat fiataloknak, időseket időseknek
ageAverage = healthDataframe['age_minmax'].mean()
ageStandardDeviation = healthDataframe['age_minmax'].std()

youngsters = healthDataframe.loc[healthDataframe['age_minmax'] < (ageAverage - (ageStandardDeviation / 2)), 'daily_steps_minmax']
oldies = healthDataframe.loc[healthDataframe['age_minmax'] > (ageAverage - (ageStandardDeviation / 2)), 'daily_steps_minmax']

# Mann–Whitney U próba
statistic, p_value = stats.mannwhitneyu(oldies, youngsters, alternative='less')
print(f"Statistics for H7: {statistic} p value for H7: {p_value}")
print("H0-at elvetjük: tehát a fiatalabbak nem feltétlen tesznek meg több lépést, mint az idősek.\n")
#p > 0.05 -> H0-at elvetjük: tehát a fiatalabbak nem feltétlen tesznek meg több lépést, mint az idősek.

#-----------------------------------------------------------------------------------------#

print("8. A nők koleszterin szintje alacsonyabb, mint a férfiaké.")
womenChLevel = healthDataframe.loc[healthDataframe['gender_Female'] == 1, 'cholesterol_minmax']
menChLevel = healthDataframe.loc[healthDataframe['gender_Male'] == 1, 'cholesterol_minmax']

# Mann–Whitney U próba
statistic, p_value = stats.mannwhitneyu(womenChLevel, menChLevel, alternative='less')
print(f"Statistics for H8: {statistic} p value for H8: {p_value}")
print("H0-at elvetjük: tehát a nők koleszterin szintje nem feltétlen alacsonyabb, mint a férfiaké\n")
#p > 0.05 -> H0-at elvetjük: tehát a nők koleszterin szintje nem feltétlen alacsonyabb, mint a férfiaké

#-----------------------------------------------------------------------------------------#

print("9. Minél több kalóriát visz be, annál kevesebbet iszik.")
# H0: a két minta lineáris korrelációban van egymással
corr, p_value = pearsonr(healthDataframe['calories_consumed_minmax'], healthDataframe['water_intake_l_minmax'])
print('Pearsons korreláció: %.3f' % corr, ', p-value =', p_value)
print("A korreláció érték gyenge, negatív korrelációt mutat, H0-t megtartjuk.\n")
#p_value < 0.05 -> H0-at megtartjuk

x = healthDataframe['calories_consumed_minmax']
y = healthDataframe['water_intake_l_minmax']
result = linregress(x, y)
print("R =", result.rvalue, ", p-value =", result.pvalue)
print("Slope: ", result.slope, result.stderr)
print("Intercept: ", result.intercept, result.intercept_stderr)

plt.plot(x, y, 'o', label='Alapadatok')
plt.plot(x, result.slope * np.array(x) + result.intercept, 'r', label='Kalória és vízbevitel közötti kapcsolat')
plt.legend()
plt.show()

#-----------------------------------------------------------------------------------------#
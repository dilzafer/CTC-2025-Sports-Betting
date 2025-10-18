import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
races = pd.read_csv('all_race_results.csv')
drivers = pd.read_csv('driver_roster.csv')

# Merge datasets
df = races.merge(drivers[['driver_name', 'aggressiveness', 'defensive_ability', 'stamina', 'age']],
                 left_on='driver', right_on='driver_name', how='left')

print("="*80)
print("AGE OPTIMIZATION ANALYSIS")
print("="*80)

# Overall age analysis
print("\n" + "-"*80)
print("AGE DISTRIBUTION IN DATASET")
print("-"*80)

print(f"\nAge range: {drivers['age'].min()} - {drivers['age'].max()} years")
print(f"Mean age: {drivers['age'].mean():.1f} years")
print(f"Median age: {drivers['age'].median():.1f} years")

# Age vs performance
print("\n" + "-"*80)
print("AGE vs RACE TIME CORRELATION")
print("-"*80)

age_corr, age_pval = stats.pearsonr(df['age'], df['time'])
print(f"\nOverall correlation: {age_corr:.4f} (p={age_pval:.6f})")
if age_pval < 0.05:
    print(f"  → Age effect is STATISTICALLY SIGNIFICANT")
    if age_corr > 0:
        print(f"  → Older drivers have SLOWER times")
    else:
        print(f"  → Older drivers have FASTER times")
else:
    print(f"  → Age effect is NOT statistically significant")

# Age by weather condition
print("\n" + "-"*80)
print("AGE EFFECT BY WEATHER CONDITION")
print("-"*80)

weather_age_effects = {}

for weather in ['Sunny', 'Wet', 'Unknown']:
    weather_df = df[df['weather'] == weather]
    corr, pval = stats.pearsonr(weather_df['age'], weather_df['time'])
    weather_age_effects[weather] = {'corr': corr, 'pval': pval}

    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"\n{weather} conditions:")
    print(f"  Correlation: {corr:7.4f} (p={pval:.6f}) {sig}")

    # Mean time by age groups
    weather_df['age_group'] = pd.cut(weather_df['age'], bins=[19, 25, 30, 35, 41],
                                      labels=['20-25', '26-30', '31-35', '36+'])
    age_group_times = weather_df.groupby('age_group')['time'].mean()
    print(f"  Average times by age group:")
    for age_grp, avg_time in age_group_times.items():
        print(f"    {age_grp}: {avg_time:.2f}s")

# Regression analysis including age
print("\n" + "="*80)
print("REGRESSION ANALYSIS: INCLUDING AGE")
print("="*80)

weather_models_with_age = {}
avg_laps = df['laps'].mean()

for weather in ['Sunny', 'Wet', 'Unknown']:
    print(f"\n{'-'*80}")
    print(f"{weather} CONDITIONS")
    print(f"{'-'*80}")

    weather_df = df[df['weather'] == weather].copy()

    X = weather_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
    y = weather_df['time'].copy()

    model = LinearRegression()
    model.fit(X, y)
    weather_models_with_age[weather] = model

    print(f"\nRegression Coefficients:")
    features = ['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']
    for i, feature in enumerate(features):
        direction = "SLOWER" if model.coef_[i] > 0 else "FASTER"
        print(f"  {feature:25s}: {model.coef_[i]:10.2f}  ({direction})")

    print(f"\n  AGE coefficient: {model.coef_[3]:.2f}s per year")
    if abs(model.coef_[3]) > 100:
        print(f"  → Age has STRONG impact")
    elif abs(model.coef_[3]) > 50:
        print(f"  → Age has MODERATE impact")
    else:
        print(f"  → Age has WEAK impact")

# Top performers age analysis
print("\n" + "="*80)
print("TOP PERFORMERS' AGE ANALYSIS")
print("="*80)

driver_performance = df.groupby('driver').agg({
    'time': ['mean', 'count'],
    'position': 'mean'
})
driver_performance.columns = ['avg_time', 'races', 'avg_position']
driver_performance = driver_performance[driver_performance['races'] >= 50]

# Top 10 by time
top10 = driver_performance.nsmallest(10, 'avg_time')

print("\nTop 10 Fastest Drivers:")
print(f"{'Rank':<6} {'Driver':<25} {'Avg Time':<12} {'Age':<6}")
print("-" * 55)

for rank, (driver, perf) in enumerate(top10.iterrows(), 1):
    driver_info = drivers[drivers['driver_name'] == driver].iloc[0]
    print(f"{rank:<6} {driver:<25} {perf['avg_time']:.2f}s      {driver_info['age']}")

top10_ages = [drivers[drivers['driver_name'] == d].iloc[0]['age'] for d in top10.index]
print(f"\nTop 10 performers average age: {np.mean(top10_ages):.1f} years")
print(f"Top 10 performers age range: {min(top10_ages)} - {max(top10_ages)} years")
print(f"Top 10 performers median age: {np.median(top10_ages):.1f} years")

# Bottom 10 by time
bottom10 = driver_performance.nlargest(10, 'avg_time')
bottom10_ages = [drivers[drivers['driver_name'] == d].iloc[0]['age'] for d in bottom10.index]

print(f"\nBottom 10 performers average age: {np.mean(bottom10_ages):.1f} years")
print(f"Difference: {np.mean(bottom10_ages) - np.mean(top10_ages):.1f} years")

# Statistical test
t_stat, t_pval = stats.ttest_ind(top10_ages, bottom10_ages)
print(f"\nT-test (top 10 vs bottom 10): t={t_stat:.3f}, p={t_pval:.4f}")
if t_pval < 0.05:
    print("  → Age difference is STATISTICALLY SIGNIFICANT")
else:
    print("  → Age difference is NOT statistically significant")

# Age winners vs losers by weather
print("\n" + "="*80)
print("WINNERS vs LOSERS AGE COMPARISON BY WEATHER")
print("="*80)

for weather in ['Sunny', 'Wet', 'Unknown']:
    weather_df = df[df['weather'] == weather]

    winners = weather_df[weather_df['position'] == 1]
    losers = weather_df[weather_df['position'] >= 15]

    winner_ages = winners['age'].values
    loser_ages = losers['age'].values

    print(f"\n{weather} Conditions:")
    print(f"  Winners avg age:        {winner_ages.mean():.1f} years (n={len(winner_ages)})")
    print(f"  Poor performers avg age: {loser_ages.mean():.1f} years (n={len(loser_ages)})")
    print(f"  Difference:             {winner_ages.mean() - loser_ages.mean():+.1f} years")

    t_stat, t_pval = stats.ttest_ind(winner_ages, loser_ages)
    sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else "ns"
    print(f"  T-test: t={t_stat:.3f}, p={t_pval:.4f} {sig}")

# Optimal age prediction
print("\n" + "="*80)
print("OPTIMAL AGE PREDICTION")
print("="*80)

# Using optimal stats from previous analysis
optimal_stats = [0.10, 0.95, 0.15]  # agg, def, stam

print(f"\nTesting optimal stats (Agg=0.10, Def=0.95, Stam=0.15) at different ages...")
print(f"\n{'Age':<6} {'Sunny':<12} {'Wet':<12} {'Unknown':<12} {'Weighted Avg':<12}")
print("-" * 60)

age_results = []

for age in range(20, 41, 2):
    times = {}
    for weather in ['Sunny', 'Wet', 'Unknown']:
        model = weather_models_with_age[weather]
        X = np.array([[optimal_stats[0], optimal_stats[1], optimal_stats[2], age, avg_laps]])
        predicted_time = model.predict(X)[0]
        times[weather] = predicted_time

    # Weighted average
    weighted = (times['Sunny'] * 0.37 + times['Wet'] * 0.27 + times['Unknown'] * 0.36)

    age_results.append({
        'age': age,
        'sunny': times['Sunny'],
        'wet': times['Wet'],
        'unknown': times['Unknown'],
        'weighted': weighted
    })

    print(f"{age:<6} {times['Sunny']:<12.2f} {times['Wet']:<12.2f} {times['Unknown']:<12.2f} {weighted:<12.2f}")

# Find optimal age
best_age_result = min(age_results, key=lambda x: x['weighted'])

print("\n" + "="*80)
print("FINAL RECOMMENDATION: OPTIMAL AGE")
print("="*80)

print(f"\n🎂 OPTIMAL AGE: {best_age_result['age']} years")

print(f"\nExpected performance at age {best_age_result['age']}:")
print(f"  Sunny:        {best_age_result['sunny']:.2f}s")
print(f"  Wet:          {best_age_result['wet']:.2f}s")
print(f"  Unknown:      {best_age_result['unknown']:.2f}s")
print(f"  Weighted Avg: {best_age_result['weighted']:.2f}s")

# Compare to other ages
worst_age_result = max(age_results, key=lambda x: x['weighted'])
age_impact = worst_age_result['weighted'] - best_age_result['age']

print(f"\n" + "-"*80)
print("AGE IMPACT ANALYSIS")
print("-"*80)

print(f"\nBest age:  {best_age_result['age']} years → {best_age_result['weighted']:.2f}s avg")
print(f"Worst age: {worst_age_result['age']} years → {worst_age_result['weighted']:.2f}s avg")
print(f"Difference: {worst_age_result['weighted'] - best_age_result['weighted']:.2f}s")

# Calculate impact relative to other stats
print(f"\n" + "-"*80)
print("RELATIVE IMPORTANCE: AGE vs OTHER ATTRIBUTES")
print("-"*80)

age_range_impact = max([r['weighted'] for r in age_results]) - min([r['weighted'] for r in age_results])
print(f"\nAge impact (20-40 years range): {age_range_impact:.2f}s")

# Compare to changing other stats by 0.1
print(f"\nFor comparison, changing stats by 0.1:")
for weather in ['Sunny', 'Wet', 'Unknown']:
    model = weather_models_with_age[weather]
    coefs = model.coef_
    print(f"  {weather}:")
    print(f"    Aggressiveness +0.1: {coefs[0]*0.1:+.2f}s")
    print(f"    Defensive -0.1:      {coefs[1]*-0.1:+.2f}s")
    print(f"    Stamina +0.1:        {coefs[2]*0.1:+.2f}s")
    print(f"    Age +10 years:       {coefs[3]*10:+.2f}s")

print(f"\n" + "-"*80)
print("KEY INSIGHTS")
print("-"*80)

if abs(age_corr) < 0.05:
    print(f"\n✓ Age has MINIMAL overall impact (correlation = {age_corr:.4f})")
    print(f"✓ Focus should remain on the three main attributes")
    print(f"✓ Optimal age: {best_age_result['age']} years (but not critical)")
elif age_corr < 0:
    print(f"\n✓ Older drivers perform BETTER (correlation = {age_corr:.4f})")
    print(f"✓ Experience matters: aim for age {best_age_result['age']}+")
else:
    print(f"\n✓ Younger drivers perform BETTER (correlation = {age_corr:.4f})")
    print(f"✓ Youth advantage: aim for age {best_age_result['age']}")

print(f"\n✓ Top performers average age: {np.mean(top10_ages):.1f} years")
print(f"✓ Current Tapped In Team age: 20.0 years")

print("\n" + "="*80)

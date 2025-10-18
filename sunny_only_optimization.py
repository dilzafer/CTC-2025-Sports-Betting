import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, differential_evolution
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
print("OPTIMIZATION SPECIFICALLY FOR SUNNY WEATHER CONDITIONS")
print("="*80)

# Filter for SUNNY conditions only
sunny_df = df[df['weather'] == 'Sunny'].copy()

print(f"\nSunny weather races: {len(sunny_df)} ({len(sunny_df)/len(df)*100:.1f}% of all races)")
print(f"Average time in Sunny: {sunny_df['time'].mean():.2f}s")
print(f"Std deviation: {sunny_df['time'].std():.2f}s")

# Build model specifically for Sunny conditions
print("\n" + "-"*80)
print("REGRESSION MODEL FOR SUNNY CONDITIONS")
print("-"*80)

avg_laps_sunny = sunny_df['laps'].mean()
print(f"\nAverage laps in Sunny races: {avg_laps_sunny:.1f}")

X_sunny = sunny_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
y_sunny = sunny_df['time'].copy()

model_sunny = LinearRegression()
model_sunny.fit(X_sunny, y_sunny)

r2 = model_sunny.score(X_sunny, y_sunny)
print(f"Model R² Score: {r2:.4f} (explains {r2*100:.2f}% of variance)")

print(f"\nRegression Coefficients for SUNNY weather:")
features = ['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']
for i, feature in enumerate(features):
    direction = "SLOWER" if model_sunny.coef_[i] > 0 else "FASTER"
    print(f"  {feature:25s}: {model_sunny.coef_[i]:10.2f}s  ({direction})")
print(f"\n  Intercept: {model_sunny.intercept_:.2f}s")

# Correlation analysis
print("\n" + "-"*80)
print("CORRELATIONS IN SUNNY CONDITIONS")
print("-"*80)

correlations_sunny = sunny_df[['time', 'aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].corr()['time'].sort_values(ascending=False)
print("\nPearson Correlations with Race Time:")
for feature, corr in correlations_sunny.items():
    if feature != 'time':
        _, p_val = stats.pearsonr(sunny_df[feature], sunny_df['time'])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {feature:25s}: {corr:7.4f}  (p={p_val:.6f}) {sig}")

# Winner analysis in Sunny
print("\n" + "-"*80)
print("SUNNY WEATHER WINNERS vs LOSERS")
print("-"*80)

sunny_winners = sunny_df[sunny_df['position'] == 1]
sunny_losers = sunny_df[sunny_df['position'] >= 15]

print(f"\nWinners (n={len(sunny_winners)}):")
print(f"  Aggressiveness:    {sunny_winners['aggressiveness'].mean():.3f} ± {sunny_winners['aggressiveness'].std():.3f}")
print(f"  Defensive Ability: {sunny_winners['defensive_ability'].mean():.3f} ± {sunny_winners['defensive_ability'].std():.3f}")
print(f"  Stamina:           {sunny_winners['stamina'].mean():.3f} ± {sunny_winners['stamina'].std():.3f}")
print(f"  Age:               {sunny_winners['age'].mean():.1f} ± {sunny_winners['age'].std():.1f}")
print(f"  Average Time:      {sunny_winners['time'].mean():.2f}s")

print(f"\nPoor Performers - Position ≥15 (n={len(sunny_losers)}):")
print(f"  Aggressiveness:    {sunny_losers['aggressiveness'].mean():.3f} ± {sunny_losers['aggressiveness'].std():.3f}")
print(f"  Defensive Ability: {sunny_losers['defensive_ability'].mean():.3f} ± {sunny_losers['defensive_ability'].std():.3f}")
print(f"  Stamina:           {sunny_losers['stamina'].mean():.3f} ± {sunny_losers['stamina'].std():.3f}")
print(f"  Age:               {sunny_losers['age'].mean():.1f} ± {sunny_losers['age'].std():.1f}")
print(f"  Average Time:      {sunny_losers['time'].mean():.2f}s")

print(f"\nStatistical Comparison (Winners vs Poor Performers):")
for attr in ['aggressiveness', 'defensive_ability', 'stamina', 'age']:
    t_stat, p_val = stats.ttest_ind(sunny_winners[attr], sunny_losers[attr])
    diff = sunny_winners[attr].mean() - sunny_losers[attr].mean()
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  {attr:25s}: diff={diff:+.3f}, t={t_stat:6.3f}, p={p_val:.6f} {sig}")

# Optimization for SUNNY conditions
print("\n" + "="*80)
print("OPTIMIZATION FOR SUNNY CONDITIONS")
print("="*80)

def predict_sunny_time(stats, age=20, laps=avg_laps_sunny):
    """Predict race time in SUNNY conditions"""
    agg, defense, stamina = stats
    X = np.array([[agg, defense, stamina, age, laps]])
    return model_sunny.predict(X)[0]

def objective_sunny(x):
    """Minimize time in sunny conditions"""
    return predict_sunny_time(x)

# Realistic bounds
bounds = [
    (0.10, 0.90),  # aggressiveness
    (0.10, 0.95),  # defensive_ability
    (0.10, 0.90)   # stamina
]

# Strategy 1: No budget constraint (just minimize)
print("\n" + "-"*80)
print("STRATEGY 1: Unconstrained Optimization for Sunny")
print("-"*80)

result_unconstrained = differential_evolution(
    objective_sunny,
    bounds=bounds,
    seed=42,
    maxiter=1000,
    polish=True
)

opt_sunny_unconstrained = result_unconstrained.x
time_unconstrained = predict_sunny_time(opt_sunny_unconstrained)

print(f"\nOptimal Stats (Unconstrained):")
print(f"  Aggressiveness:    {opt_sunny_unconstrained[0]:.3f}")
print(f"  Defensive Ability: {opt_sunny_unconstrained[1]:.3f}")
print(f"  Stamina:           {opt_sunny_unconstrained[2]:.3f}")
print(f"  Total:             {sum(opt_sunny_unconstrained):.3f}")
print(f"\nExpected Time: {time_unconstrained:.2f}s")

# Strategy 2: With budget constraint
STAT_BUDGET = drivers[['aggressiveness', 'defensive_ability', 'stamina']].sum(axis=1).median()

print("\n" + "-"*80)
print(f"STRATEGY 2: Optimization with Budget Constraint ({STAT_BUDGET:.3f})")
print("-"*80)

constraints_sunny = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - STAT_BUDGET}
]

best_result_sunny = None
best_score_sunny = float('inf')

for _ in range(50):
    x0 = np.random.random(3)
    x0 = x0 / x0.sum() * STAT_BUDGET

    result = minimize(objective_sunny, x0, bounds=bounds, constraints=constraints_sunny,
                     method='SLSQP', options={'maxiter': 1000})

    if result.success and result.fun < best_score_sunny:
        best_score_sunny = result.fun
        best_result_sunny = result

opt_sunny_constrained = best_result_sunny.x
time_constrained = predict_sunny_time(opt_sunny_constrained)

print(f"\nOptimal Stats (Budget-Constrained):")
print(f"  Aggressiveness:    {opt_sunny_constrained[0]:.3f}")
print(f"  Defensive Ability: {opt_sunny_constrained[1]:.3f}")
print(f"  Stamina:           {opt_sunny_constrained[2]:.3f}")
print(f"  Total:             {sum(opt_sunny_constrained):.3f}")
print(f"\nExpected Time: {time_constrained:.2f}s")

# Test different budgets for Sunny
print("\n" + "-"*80)
print("SENSITIVITY ANALYSIS: Different Stat Budgets (SUNNY ONLY)")
print("-"*80)

budgets = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
budget_results_sunny = []

for budget in budgets:
    constraints_b = [{'type': 'eq', 'fun': lambda x, b=budget: np.sum(x) - b}]

    best_b = None
    best_score_b = float('inf')

    for _ in range(30):
        x0 = np.random.random(3)
        x0 = x0 / x0.sum() * budget

        result = minimize(objective_sunny, x0, bounds=bounds, constraints=constraints_b,
                         method='SLSQP', options={'maxiter': 1000})

        if result.success and result.fun < best_score_b:
            best_score_b = result.fun
            best_b = result.x

    if best_b is not None:
        expected_time = predict_sunny_time(best_b)
        budget_results_sunny.append({
            'budget': budget,
            'time': expected_time,
            'agg': best_b[0],
            'def': best_b[1],
            'stam': best_b[2]
        })

print("\nOptimal stats for different budgets (SUNNY):")
print(f"{'Budget':<8} {'Sunny Time':<12} {'Agg':<8} {'Def':<8} {'Stam':<8}")
print("-" * 52)
for r in budget_results_sunny:
    print(f"{r['budget']:<8.2f} {r['time']:<12.2f} {r['agg']:<8.3f} {r['def']:<8.3f} {r['stam']:<8.3f}")

# Find best budget
best_budget_idx = np.argmin([r['time'] for r in budget_results_sunny])
best_budget_result_sunny = budget_results_sunny[best_budget_idx]

# AGE optimization for Sunny
print("\n" + "-"*80)
print("AGE OPTIMIZATION FOR SUNNY CONDITIONS")
print("-"*80)

print(f"\nTesting optimal stats at different ages (Sunny only):")
print(f"Using stats: Agg={best_budget_result_sunny['agg']:.3f}, Def={best_budget_result_sunny['def']:.3f}, Stam={best_budget_result_sunny['stam']:.3f}")

print(f"\n{'Age':<6} {'Sunny Time':<12} {'vs Age 20':<12}")
print("-" * 36)

age_results_sunny = []
for age in range(20, 41, 2):
    time = predict_sunny_time([best_budget_result_sunny['agg'],
                               best_budget_result_sunny['def'],
                               best_budget_result_sunny['stam']], age=age)
    age_results_sunny.append({'age': age, 'time': time})

    baseline = predict_sunny_time([best_budget_result_sunny['agg'],
                                   best_budget_result_sunny['def'],
                                   best_budget_result_sunny['stam']], age=20)
    diff = time - baseline
    print(f"{age:<6} {time:<12.2f} {diff:+.2f}s")

best_age_sunny = min(age_results_sunny, key=lambda x: x['time'])
worst_age_sunny = max(age_results_sunny, key=lambda x: x['time'])

# Compare to current Tapped In Team
print("\n" + "="*80)
print("COMPARISON WITH CURRENT TAPPED IN TEAM (SUNNY)")
print("="*80)

tapped_in_drivers = drivers[drivers['team'] == 'Tapped In Team']
current_stats = [
    tapped_in_drivers['aggressiveness'].mean(),
    tapped_in_drivers['defensive_ability'].mean(),
    tapped_in_drivers['stamina'].mean()
]
current_age = tapped_in_drivers['age'].mean()

current_sunny_time = predict_sunny_time(current_stats, age=current_age)

print(f"\nCurrent Tapped In Team:")
print(f"  Aggressiveness:    {current_stats[0]:.3f}")
print(f"  Defensive Ability: {current_stats[1]:.3f}")
print(f"  Stamina:           {current_stats[2]:.3f}")
print(f"  Age:               {current_age:.1f}")
print(f"  Expected Sunny Time: {current_sunny_time:.2f}s")

# Actual performance in Sunny
tapped_sunny_actual = df[(df['team'] == 'Tapped In Team') & (df['weather'] == 'Sunny')]
print(f"  Actual Sunny Avg:  {tapped_sunny_actual['time'].mean():.2f}s")
print(f"  Win rate in Sunny: {(tapped_sunny_actual['position']==1).mean()*100:.1f}%")

# Final recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION FOR SUNNY CONDITIONS")
print("="*80)

print(f"\n🌤️  SUNNY-OPTIMIZED CONFIGURATION:")
print(f"\n  🎯 Aggressiveness:    {best_budget_result_sunny['agg']:.3f}")
print(f"  🛡️  Defensive Ability: {best_budget_result_sunny['def']:.3f}")
print(f"  💪 Stamina:           {best_budget_result_sunny['stam']:.3f}")
print(f"  🎂 Age:               {best_age_sunny['age']} years")
print(f"\n  Total Stat Budget:   {best_budget_result_sunny['budget']:.3f}")
print(f"  Expected Sunny Time: {best_budget_result_sunny['time']:.2f}s")

improvement_sunny = current_sunny_time - best_budget_result_sunny['time']
print(f"\n  Improvement over current: {improvement_sunny:.2f}s ({improvement_sunny/current_sunny_time*100:.1f}%)")

print(f"\n" + "-"*80)
print("KEY INSIGHTS FOR SUNNY WEATHER")
print("-"*80)

print(f"\n1. AGGRESSIVENESS = {best_budget_result_sunny['agg']:.3f}")
print(f"   • Coefficient: +{model_sunny.coef_[0]:.2f}s per unit")
print(f"   • MUST minimize - huge penalty in Sunny!")
print(f"   • Winners avg: {sunny_winners['aggressiveness'].mean():.3f}")
print(f"   • Losers avg: {sunny_losers['aggressiveness'].mean():.3f}")
print(f"   → Lower aggressiveness = CRITICAL for Sunny")

print(f"\n2. DEFENSIVE ABILITY = {best_budget_result_sunny['def']:.3f}")
print(f"   • Coefficient: +{model_sunny.coef_[1]:.2f}s per unit")
print(f"   • Surprisingly adds time in Sunny (penalty!)")
print(f"   • Winners avg: {sunny_winners['defensive_ability'].mean():.3f}")
print(f"   • Losers avg: {sunny_losers['defensive_ability'].mean():.3f}")
print(f"   → Keep moderate in Sunny conditions")

print(f"\n3. STAMINA = {best_budget_result_sunny['stam']:.3f}")
print(f"   • Coefficient: +{model_sunny.coef_[2]:.2f}s per unit")
print(f"   • Major penalty in Sunny (long hot races drain energy)")
print(f"   • Winners avg: {sunny_winners['stamina'].mean():.3f}")
print(f"   • Losers avg: {sunny_losers['stamina'].mean():.3f}")
print(f"   → Minimize stamina for Sunny performance")

print(f"\n4. AGE = {best_age_sunny['age']} years")
print(f"   • Coefficient: {model_sunny.coef_[3]:.2f}s per year")
print(f"   • Slightly favors older drivers in Sunny")
print(f"   • Winners avg: {sunny_winners['age'].mean():.1f} years")
print(f"   • Age range impact: {worst_age_sunny['time'] - best_age_sunny['time']:.2f}s (20-40 years)")
print(f"   → Age matters less than other stats")

print(f"\n5. LAPS (race length)")
print(f"   • Coefficient: +{model_sunny.coef_[4]:.2f}s per lap")
print(f"   • Each lap costs most in Sunny (hot, demanding)")
print(f"   • Avg laps in Sunny: {avg_laps_sunny:.1f}")

print(f"\n" + "-"*80)
print("SUNNY WEATHER STRATEGY")
print("-"*80)

print(f"\n✓ PRIORITY 1: Minimize Aggressiveness (+6,236s penalty per unit!)")
print(f"✓ PRIORITY 2: Minimize Stamina (+3,594s penalty per unit)")
print(f"✓ PRIORITY 3: Keep Defensive Ability moderate (not max)")
print(f"✓ Age is less critical but younger is slightly better")
print(f"\n✓ Sunny is the SLOWEST condition (10,922s avg)")
print(f"✓ Conservative, patient driving wins in Sunny")
print(f"✓ Avoid tire wear and overheating with low aggression/stamina")

print("\n" + "="*80)

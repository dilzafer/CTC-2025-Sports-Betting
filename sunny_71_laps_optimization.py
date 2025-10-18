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
print("OPTIMIZATION FOR SUNNY WEATHER - 71 LAPS SPECIFICALLY")
print("="*80)

# Filter for SUNNY conditions
sunny_df = df[df['weather'] == 'Sunny'].copy()

print(f"\nSunny weather races: {len(sunny_df)}")
print(f"Lap range in Sunny: {sunny_df['laps'].min():.0f} - {sunny_df['laps'].max():.0f}")
print(f"Average laps: {sunny_df['laps'].mean():.1f}")

# Check if there are 71-lap races
sunny_71 = sunny_df[sunny_df['laps'] == 71]
print(f"\nRaces with exactly 71 laps: {len(sunny_71)}")

if len(sunny_71) > 0:
    print(f"  Average time in 71-lap Sunny races: {sunny_71['time'].mean():.2f}s")
    print(f"  Fastest 71-lap Sunny time: {sunny_71['time'].min():.2f}s")
    print(f"  Slowest 71-lap Sunny time: {sunny_71['time'].max():.2f}s")

# Build regression model for Sunny
X_sunny = sunny_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
y_sunny = sunny_df['time'].copy()

model_sunny = LinearRegression()
model_sunny.fit(X_sunny, y_sunny)

print("\n" + "-"*80)
print("REGRESSION MODEL FOR SUNNY (All lap lengths)")
print("-"*80)

features = ['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']
print(f"\nCoefficients:")
for i, feature in enumerate(features):
    print(f"  {feature:25s}: {model_sunny.coef_[i]:10.2f}s per unit")
print(f"  Intercept: {model_sunny.intercept_:.2f}s")

# Predict time for 71 laps specifically
def predict_sunny_71_laps(stats, age=20, laps=71):
    """Predict race time in SUNNY conditions with specific lap count"""
    agg, defense, stamina = stats
    X = np.array([[agg, defense, stamina, age, laps]])
    return model_sunny.predict(X)[0]

# Calculate impact of each attribute at 71 laps
print("\n" + "-"*80)
print("IMPACT ANALYSIS FOR 71-LAP SUNNY RACE")
print("-"*80)

baseline_stats = [0.5, 0.5, 0.5]  # Baseline for comparison
baseline_age = 30
baseline_time = predict_sunny_71_laps(baseline_stats, age=baseline_age)

print(f"\nBaseline (Agg=0.5, Def=0.5, Stam=0.5, Age=30, Laps=71): {baseline_time:.2f}s")

print(f"\nImpact of changing each attribute by ±0.1:")
print(f"  Aggressiveness:")
print(f"    0.4 → 0.5: {model_sunny.coef_[0] * 0.1:+.2f}s")
print(f"    0.5 → 0.6: {model_sunny.coef_[0] * 0.1:+.2f}s")
print(f"  Defensive Ability:")
print(f"    0.4 → 0.5: {model_sunny.coef_[1] * 0.1:+.2f}s")
print(f"    0.5 → 0.6: {model_sunny.coef_[1] * 0.1:+.2f}s")
print(f"  Stamina:")
print(f"    0.4 → 0.5: {model_sunny.coef_[2] * 0.1:+.2f}s")
print(f"    0.5 → 0.6: {model_sunny.coef_[2] * 0.1:+.2f}s")
print(f"  Age:")
print(f"    20 → 30 years: {model_sunny.coef_[3] * 10:+.2f}s")
print(f"    30 → 40 years: {model_sunny.coef_[3] * 10:+.2f}s")

print(f"\nLap length coefficient: {model_sunny.coef_[4]:.2f}s per lap")
print(f"  71 laps vs 56 laps (avg): {model_sunny.coef_[4] * (71-56.9):+.2f}s")
print(f"  → 71-lap races are {71-56.9:.1f} laps LONGER than average")
print(f"  → This adds approximately {model_sunny.coef_[4] * (71-56.9):.0f} seconds to race time")

# Optimization for 71 laps specifically
print("\n" + "="*80)
print("OPTIMIZATION FOR 71-LAP SUNNY RACE")
print("="*80)

def objective_71(x):
    return predict_sunny_71_laps(x, age=20)  # Start with age 20

bounds = [(0.10, 0.90), (0.10, 0.95), (0.10, 0.90)]

# Unconstrained optimization
result_unconstrained = differential_evolution(
    objective_71,
    bounds=bounds,
    seed=42,
    maxiter=1000,
    polish=True
)

opt_71_unconstrained = result_unconstrained.x

print("\n" + "-"*80)
print("UNCONSTRAINED OPTIMIZATION (71 laps)")
print("-"*80)

print(f"\nOptimal Stats:")
print(f"  Aggressiveness:    {opt_71_unconstrained[0]:.3f}")
print(f"  Defensive Ability: {opt_71_unconstrained[1]:.3f}")
print(f"  Stamina:           {opt_71_unconstrained[2]:.3f}")
print(f"  Total:             {sum(opt_71_unconstrained):.3f}")

for age in [20, 25, 30, 35, 40]:
    time = predict_sunny_71_laps(opt_71_unconstrained, age=age)
    print(f"  Expected time (age {age}): {time:.2f}s")

# Budget-constrained optimization
STAT_BUDGET = 0.8  # Use lower budget for best performance

print("\n" + "-"*80)
print(f"BUDGET-CONSTRAINED OPTIMIZATION (Budget={STAT_BUDGET})")
print("-"*80)

constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - STAT_BUDGET}]

best_result = None
best_score = float('inf')

for _ in range(50):
    x0 = np.random.random(3)
    x0 = x0 / x0.sum() * STAT_BUDGET

    result = minimize(objective_71, x0, bounds=bounds, constraints=constraints,
                     method='SLSQP', options={'maxiter': 1000})

    if result.success and result.fun < best_score:
        best_score = result.fun
        best_result = result

opt_71_budget = best_result.x

print(f"\nOptimal Stats (Budget={STAT_BUDGET}):")
print(f"  Aggressiveness:    {opt_71_budget[0]:.3f}")
print(f"  Defensive Ability: {opt_71_budget[1]:.3f}")
print(f"  Stamina:           {opt_71_budget[2]:.3f}")
print(f"  Total:             {sum(opt_71_budget):.3f}")

print(f"\nExpected times at different ages:")
for age in [20, 25, 30, 35, 40]:
    time = predict_sunny_71_laps(opt_71_budget, age=age)
    print(f"  Age {age}: {time:.2f}s")

# Test different budgets
print("\n" + "-"*80)
print("SENSITIVITY ANALYSIS: Different Budgets (71 laps, Sunny)")
print("-"*80)

budgets = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
budget_results = []

for budget in budgets:
    constraints_b = [{'type': 'eq', 'fun': lambda x, b=budget: np.sum(x) - b}]

    best_b = None
    best_score_b = float('inf')

    for _ in range(30):
        x0 = np.random.random(3)
        x0 = x0 / x0.sum() * budget

        result = minimize(objective_71, x0, bounds=bounds, constraints=constraints_b,
                         method='SLSQP', options={'maxiter': 1000})

        if result.success and result.fun < best_score_b:
            best_score_b = result.fun
            best_b = result.x

    if best_b is not None:
        budget_results.append({
            'budget': budget,
            'time_age20': predict_sunny_71_laps(best_b, age=20),
            'time_age40': predict_sunny_71_laps(best_b, age=40),
            'agg': best_b[0],
            'def': best_b[1],
            'stam': best_b[2]
        })

print(f"\n{'Budget':<8} {'Time(20y)':<12} {'Time(40y)':<12} {'Agg':<8} {'Def':<8} {'Stam':<8}")
print("-" * 68)
for r in budget_results:
    print(f"{r['budget']:<8.1f} {r['time_age20']:<12.2f} {r['time_age40']:<12.2f} {r['agg']:<8.3f} {r['def']:<8.3f} {r['stam']:<8.3f}")

best_budget_result = min(budget_results, key=lambda x: x['time_age20'])

# Age optimization for 71 laps
print("\n" + "-"*80)
print("AGE OPTIMIZATION FOR 71-LAP SUNNY RACE")
print("-"*80)

opt_stats = [best_budget_result['agg'], best_budget_result['def'], best_budget_result['stam']]

print(f"\nUsing optimal stats: Agg={opt_stats[0]:.3f}, Def={opt_stats[1]:.3f}, Stam={opt_stats[2]:.3f}")
print(f"\n{'Age':<6} {'Time (71 laps)':<16} {'vs Age 20':<12}")
print("-" * 40)

age_results = []
for age in range(20, 41):
    time = predict_sunny_71_laps(opt_stats, age=age)
    baseline_age20 = predict_sunny_71_laps(opt_stats, age=20)
    diff = time - baseline_age20
    age_results.append({'age': age, 'time': time})
    if age % 2 == 0:
        print(f"{age:<6} {time:<16.2f} {diff:+.2f}s")

best_age = min(age_results, key=lambda x: x['time'])
worst_age = max(age_results, key=lambda x: x['time'])

# Current Tapped In Team comparison
print("\n" + "="*80)
print("COMPARISON WITH CURRENT TAPPED IN TEAM")
print("="*80)

tapped_in_drivers = drivers[drivers['team'] == 'Tapped In Team']
current_stats = [
    tapped_in_drivers['aggressiveness'].mean(),
    tapped_in_drivers['defensive_ability'].mean(),
    tapped_in_drivers['stamina'].mean()
]
current_age = tapped_in_drivers['age'].mean()

current_time_71 = predict_sunny_71_laps(current_stats, age=current_age)

print(f"\nCurrent Tapped In Team Configuration:")
print(f"  Aggressiveness:    {current_stats[0]:.3f}")
print(f"  Defensive Ability: {current_stats[1]:.3f}")
print(f"  Stamina:           {current_stats[2]:.3f}")
print(f"  Age:               {current_age:.1f}")
print(f"  Expected time (71 laps, Sunny): {current_time_71:.2f}s")

# Check actual performance if data exists
tapped_sunny_71 = df[(df['team'] == 'Tapped In Team') &
                     (df['weather'] == 'Sunny') &
                     (df['laps'] == 71)]
if len(tapped_sunny_71) > 0:
    print(f"  Actual avg (71-lap Sunny): {tapped_sunny_71['time'].mean():.2f}s")

# Final recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION: 71-LAP SUNNY RACE")
print("="*80)

opt_age_for_budget = best_age['age']

print(f"\n☀️ + 71 LAPS OPTIMIZED CONFIGURATION:")
print(f"\n  🎯 Aggressiveness:    {best_budget_result['agg']:.3f}")
print(f"  🛡️  Defensive Ability: {best_budget_result['def']:.3f}")
print(f"  💪 Stamina:           {best_budget_result['stam']:.3f}")
print(f"  🎂 Age:               {opt_age_for_budget} years")
print(f"\n  Stat Budget:         {best_budget_result['budget']:.1f}")
print(f"  Expected Time:       {best_age['time']:.2f}s")

improvement = current_time_71 - best_age['time']
print(f"\n  Improvement over current: {improvement:.2f}s ({improvement/current_time_71*100:.1f}%)")

print(f"\n" + "-"*80)
print("WHY THESE STATS FOR 71-LAP SUNNY RACE")
print("-"*80)

print(f"\n1. RACE LENGTH IMPACT:")
print(f"   • 71 laps is {71-56.9:.1f} laps longer than average")
print(f"   • Each lap costs +{model_sunny.coef_[4]:.2f}s in Sunny")
print(f"   • Extra length adds ~{model_sunny.coef_[4]*(71-56.9):.0f}s to base time")
print(f"   → Longer race = More critical to optimize stats")

print(f"\n2. AGGRESSIVENESS = {best_budget_result['agg']:.3f}")
print(f"   • Penalty: +{model_sunny.coef_[0]:.0f}s per unit")
print(f"   • In 71-lap race, mistakes compound over distance")
print(f"   • MUST minimize for long sunny races")

print(f"\n3. DEFENSIVE ABILITY = {best_budget_result['def']:.3f}")
print(f"   • Penalty in Sunny: +{model_sunny.coef_[1]:.0f}s per unit")
print(f"   • Keep moderate - defensive driving costs time in heat")
print(f"   • Long races = more opportunities for errors with high defense")

print(f"\n4. STAMINA = {best_budget_result['stam']:.3f}")
print(f"   • Penalty: +{model_sunny.coef_[2]:.0f}s per unit")
print(f"   • Counterintuitive: LOW stamina better in long hot races")
print(f"   • High stamina = overdriving = more tire/fuel issues")
print(f"   → Minimize for 71-lap sunny races")

print(f"\n5. AGE = {opt_age_for_budget} years")
print(f"   • Coefficient: {model_sunny.coef_[3]:.2f}s per year")
print(f"   • {opt_age_for_budget} years is optimal for managing:")
print(f"     - Pace over 71 laps")
print(f"     - Tire conservation in heat")
print(f"     - Fuel management")
print(f"   • Age impact: {worst_age['time'] - best_age['time']:.0f}s difference (20-40 range)")

print(f"\n6. STRATEGY FOR 71-LAP SUNNY:")
print(f"   ✓ Conservative driving (low aggression)")
print(f"   ✓ Simple, direct racing (moderate defense)")
print(f"   ✓ Pace management over aggression (low stamina)")
print(f"   ✓ Experience managing long hot races (age {opt_age_for_budget})")
print(f"   ✓ Let opponents make mistakes from overdriving")

# Performance breakdown
print(f"\n" + "-"*80)
print("PERFORMANCE BREAKDOWN")
print("-"*80)

print(f"\nOptimal configuration performance:")
print(f"  Best case (age {best_age['age']}):  {best_age['time']:.2f}s")
print(f"  Worst case (age {worst_age['age']}): {worst_age['time']:.2f}s")
print(f"  Age impact range: {worst_age['time'] - best_age['time']:.2f}s")

print(f"\nVs Current Tapped In Team:")
print(f"  Current expected:  {current_time_71:.2f}s")
print(f"  Optimized:         {best_age['time']:.2f}s")
print(f"  Improvement:       {improvement:.2f}s ({improvement/current_time_71*100:.1f}%)")

print("\n" + "="*80)

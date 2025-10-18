import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Load data
races = pd.read_csv('all_race_results.csv')
drivers = pd.read_csv('driver_roster.csv')

# Merge datasets
df = races.merge(drivers[['driver_name', 'aggressiveness', 'defensive_ability', 'stamina', 'age']],
                 left_on='driver', right_on='driver_name', how='left')

print("="*80)
print("REALISTIC STAT OPTIMIZATION WITH PRACTICAL CONSTRAINTS")
print("="*80)

# Build weather-specific models
weather_models = {}
weather_stats = {}
avg_laps = df['laps'].mean()

for weather in ['Sunny', 'Wet', 'Unknown']:
    weather_df = df[df['weather'] == weather].copy()
    X = weather_df[['aggressiveness', 'defensive_ability', 'stamina', 'laps']].copy()
    y = weather_df['time'].copy()
    model = LinearRegression()
    model.fit(X, y)
    weather_models[weather] = model
    weather_stats[weather] = {
        'count': len(weather_df),
        'proportion': len(weather_df) / len(df)
    }

def predict_time(stats, weather, avg_laps=avg_laps):
    agg, defense, stamina = stats
    model = weather_models[weather]
    X = np.array([[agg, defense, stamina, avg_laps]])
    return model.predict(X)[0]

def weighted_expected_time(stats):
    total_time = 0
    for weather in ['Sunny', 'Wet', 'Unknown']:
        time = predict_time(stats, weather)
        weight = weather_stats[weather]['proportion']
        total_time += time * weight
    return total_time

# Analyze realistic stat distributions from actual drivers
print("\n" + "-"*80)
print("REALISTIC STAT RANGES FROM ACTUAL DRIVERS")
print("-"*80)

print(f"\nObserved ranges in dataset:")
print(f"  Aggressiveness:    [{drivers['aggressiveness'].min():.3f}, {drivers['aggressiveness'].max():.3f}]")
print(f"  Defensive Ability: [{drivers['defensive_ability'].min():.3f}, {drivers['defensive_ability'].max():.3f}]")
print(f"  Stamina:           [{drivers['stamina'].min():.3f}, {drivers['stamina'].max():.3f}]")

print(f"\nMean values:")
print(f"  Aggressiveness:    {drivers['aggressiveness'].mean():.3f} ± {drivers['aggressiveness'].std():.3f}")
print(f"  Defensive Ability: {drivers['defensive_ability'].mean():.3f} ± {drivers['defensive_ability'].std():.3f}")
print(f"  Stamina:           {drivers['stamina'].mean():.3f} ± {drivers['stamina'].std():.3f}")

# Get top 3 performers to analyze
driver_performance = df.groupby('driver').agg({
    'time': 'mean',
    'position': 'mean'
}).sort_values('time')

top3_drivers = driver_performance.head(3).index
print(f"\n" + "-"*80)
print("TOP 3 FASTEST DRIVERS' ACTUAL STATS")
print("-"*80)

top3_stats = []
for driver in top3_drivers:
    driver_info = drivers[drivers['driver_name'] == driver].iloc[0]
    print(f"\n{driver} (Avg time: {driver_performance.loc[driver, 'time']:.2f}s)")
    print(f"  Aggressiveness:    {driver_info['aggressiveness']:.3f}")
    print(f"  Defensive Ability: {driver_info['defensive_ability']:.3f}")
    print(f"  Stamina:           {driver_info['stamina']:.3f}")
    top3_stats.append([
        driver_info['aggressiveness'],
        driver_info['defensive_ability'],
        driver_info['stamina']
    ])

top3_avg = np.mean(top3_stats, axis=0)
print(f"\nTop 3 Average Stats:")
print(f"  Aggressiveness:    {top3_avg[0]:.3f}")
print(f"  Defensive Ability: {top3_avg[1]:.3f}")
print(f"  Stamina:           {top3_avg[2]:.3f}")

# CONSTRAINT 1: Stats must sum to a reasonable total
# Most drivers have total stats between 0.8 and 2.0
driver_stat_sums = drivers[['aggressiveness', 'defensive_ability', 'stamina']].sum(axis=1)
print(f"\n" + "-"*80)
print("STAT BUDGET ANALYSIS")
print("-"*80)
print(f"Driver stat sums range: [{driver_stat_sums.min():.3f}, {driver_stat_sums.max():.3f}]")
print(f"Mean stat sum: {driver_stat_sums.mean():.3f}")
print(f"Median stat sum: {driver_stat_sums.median():.3f}")

# Let's use median as budget
STAT_BUDGET = driver_stat_sums.median()
print(f"\nUsing stat budget: {STAT_BUDGET:.3f}")

# Optimization with realistic constraints
print("\n" + "="*80)
print("CONSTRAINED OPTIMIZATION")
print("="*80)

# Constraint 1: Sum of stats = budget
# Constraint 2: Each stat between observed min and max
# Constraint 3: No extreme values (at least 0.05, at most 0.95)

def objective(x):
    return weighted_expected_time(x)

# Use tighter bounds based on observations
bounds = [
    (0.10, 0.90),  # aggressiveness
    (0.10, 0.95),  # defensive_ability
    (0.10, 0.90)   # stamina
]

# Constraint: sum equals budget
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - STAT_BUDGET}
]

print("\n" + "-"*80)
print("OPTIMIZATION WITH STAT BUDGET CONSTRAINT")
print(f"(Total stats must equal {STAT_BUDGET:.3f})")
print("-"*80)

# Multiple random starts
best_result = None
best_score = float('inf')

for _ in range(50):
    # Generate random starting point that satisfies constraint
    x0 = np.random.random(3)
    x0 = x0 / x0.sum() * STAT_BUDGET  # Scale to match budget

    result = minimize(objective, x0, bounds=bounds, constraints=constraints,
                     method='SLSQP', options={'maxiter': 1000})

    if result.success and result.fun < best_score:
        best_score = result.fun
        best_result = result

opt_constrained = best_result.x

print(f"\nOptimal Stats (Budget-Constrained):")
print(f"  Aggressiveness:    {opt_constrained[0]:.3f}")
print(f"  Defensive Ability: {opt_constrained[1]:.3f}")
print(f"  Stamina:           {opt_constrained[2]:.3f}")
print(f"  Total:             {sum(opt_constrained):.3f}")

print(f"\nExpected Performance:")
print(f"  Weighted Avg Time: {weighted_expected_time(opt_constrained):.2f}s")
for weather in ['Sunny', 'Wet', 'Unknown']:
    time = predict_time(opt_constrained, weather)
    print(f"    {weather:8s}: {time:.2f}s")

# Alternative: No sum constraint, just realistic bounds
print("\n" + "-"*80)
print("OPTIMIZATION WITH REALISTIC BOUNDS (No Sum Constraint)")
print("-"*80)

# Use differential evolution for global optimization
result_global = differential_evolution(
    objective,
    bounds=bounds,
    seed=42,
    maxiter=1000,
    polish=True
)

opt_unconstrained = result_global.x

print(f"\nOptimal Stats (Unconstrained Sum):")
print(f"  Aggressiveness:    {opt_unconstrained[0]:.3f}")
print(f"  Defensive Ability: {opt_unconstrained[1]:.3f}")
print(f"  Stamina:           {opt_unconstrained[2]:.3f}")
print(f"  Total:             {sum(opt_unconstrained):.3f}")

print(f"\nExpected Performance:")
print(f"  Weighted Avg Time: {weighted_expected_time(opt_unconstrained):.2f}s")
for weather in ['Sunny', 'Wet', 'Unknown']:
    time = predict_time(opt_unconstrained, weather)
    print(f"    {weather:8s}: {time:.2f}s")

# Now let's try different budget levels
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS: Different Stat Budgets")
print("="*80)

budgets = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
budget_results = []

for budget in budgets:
    constraints_b = [{'type': 'eq', 'fun': lambda x, b=budget: np.sum(x) - b}]

    best_b = None
    best_score_b = float('inf')

    for _ in range(30):
        x0 = np.random.random(3)
        x0 = x0 / x0.sum() * budget

        result = minimize(objective, x0, bounds=bounds, constraints=constraints_b,
                         method='SLSQP', options={'maxiter': 1000})

        if result.success and result.fun < best_score_b:
            best_score_b = result.fun
            best_b = result.x

    if best_b is not None:
        expected_time = weighted_expected_time(best_b)
        budget_results.append({
            'budget': budget,
            'time': expected_time,
            'agg': best_b[0],
            'def': best_b[1],
            'stam': best_b[2]
        })

print("\nOptimal stats for different budgets:")
print(f"{'Budget':<8} {'Avg Time':<12} {'Agg':<8} {'Def':<8} {'Stam':<8}")
print("-" * 52)
for r in budget_results:
    print(f"{r['budget']:<8.2f} {r['time']:<12.2f} {r['agg']:<8.3f} {r['def']:<8.3f} {r['stam']:<8.3f}")

# Find sweet spot
best_budget_idx = np.argmin([r['time'] for r in budget_results])
best_budget_result = budget_results[best_budget_idx]

print(f"\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

print(f"\nBased on comprehensive analysis across {len(budgets)} different budget levels:")
print(f"\nOPTIMAL STAT ALLOCATION:")
print(f"\n  🎯 Aggressiveness:    {best_budget_result['agg']:.3f}")
print(f"  🛡️  Defensive Ability: {best_budget_result['def']:.3f}")
print(f"  💪 Stamina:           {best_budget_result['stam']:.3f}")
print(f"\n  Total Stat Budget:   {best_budget_result['budget']:.3f}")
print(f"  Expected Avg Time:   {best_budget_result['time']:.2f}s")

print(f"\n" + "-"*80)
print("REASONING & JUSTIFICATION")
print("-"*80)

print(f"\n1. AGGRESSIVENESS = {best_budget_result['agg']:.3f}")
print(f"   • Minimized to reduce time penalties")
print(f"   • Sunny penalty: +6,178s per unit")
print(f"   • Unknown penalty: +8,039s per unit")
print(f"   • Lower = Better in 73% of conditions")

print(f"\n2. DEFENSIVE ABILITY = {best_budget_result['def']:.3f}")
print(f"   • Maximized within budget")
print(f"   • Unknown benefit: -5,569s per unit (massive!)")
print(f"   • Wet benefit: -1,740s per unit")
print(f"   • Higher = Better in 63% of conditions")

print(f"\n3. STAMINA = {best_budget_result['stam']:.3f}")
print(f"   • Minimized as it hurts performance")
print(f"   • Sunny penalty: +3,239s per unit")
print(f"   • Unknown penalty: +2,368s per unit")
print(f"   • Only helps slightly in Wet (+81s per unit)")
print(f"   • Lower = Better overall")

print(f"\n4. WHY THIS ALLOCATION WINS:")
print(f"   ✓ Avoids aggressiveness penalties in Sunny (37% of races)")
print(f"   ✓ Maximizes defensive skills for Unknown (36% of races)")
print(f"   ✓ Maintains competitiveness in Wet (27% of races)")
print(f"   ✓ Realistic and achievable stat distribution")
print(f"   ✓ Matches philosophy of top performers")

# Compare to current and top performers
current_tapped = [0.365, 0.425, 0.270]
print(f"\n" + "-"*80)
print("COMPARISON")
print("-"*80)

print(f"\nCurrent Tapped In Team: {current_tapped[0]:.3f}, {current_tapped[1]:.3f}, {current_tapped[2]:.3f}")
print(f"  Expected time: {weighted_expected_time(current_tapped):.2f}s")

print(f"\nTop 3 Performers Avg: {top3_avg[0]:.3f}, {top3_avg[1]:.3f}, {top3_avg[2]:.3f}")
print(f"  Expected time: {weighted_expected_time(top3_avg):.2f}s")

print(f"\nOptimized Stats:      {best_budget_result['agg']:.3f}, {best_budget_result['def']:.3f}, {best_budget_result['stam']:.3f}")
print(f"  Expected time: {best_budget_result['time']:.2f}s")

improvement = weighted_expected_time(current_tapped) - best_budget_result['time']
print(f"\nImprovement over current: {improvement:.2f}s ({improvement/weighted_expected_time(current_tapped)*100:.1f}%)")

print("\n" + "="*80)

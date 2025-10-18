import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Load data
races = pd.read_csv('all_race_results.csv')
drivers = pd.read_csv('driver_roster.csv')

# Merge datasets
df = races.merge(drivers[['driver_name', 'aggressiveness', 'defensive_ability', 'stamina', 'age']],
                 left_on='driver', right_on='driver_name', how='left')

print("="*80)
print("OPTIMAL STAT ALLOCATION ANALYSIS FOR TAPPED IN TEAM")
print("="*80)

# Analyze current Tapped In Team drivers
print("\n" + "-"*80)
print("CURRENT TAPPED IN TEAM DRIVERS")
print("-"*80)

tapped_in_drivers = drivers[drivers['team'] == 'Tapped In Team']
print(tapped_in_drivers[['driver_name', 'aggressiveness', 'defensive_ability', 'stamina', 'age']])

print("\nCurrent Team Averages:")
print(f"  Aggressiveness:    {tapped_in_drivers['aggressiveness'].mean():.3f}")
print(f"  Defensive Ability: {tapped_in_drivers['defensive_ability'].mean():.3f}")
print(f"  Stamina:           {tapped_in_drivers['stamina'].mean():.3f}")
print(f"  Age:               {tapped_in_drivers['age'].mean():.1f}")

# Their performance
tapped_in_races = df[df['team'] == 'Tapped In Team']
print(f"\nCurrent Performance:")
print(f"  Average Time:      {tapped_in_races['time'].mean():.2f}s")
print(f"  Win Rate:          {(tapped_in_races['position']==1).mean()*100:.2f}%")
print(f"  Podium Rate:       {(tapped_in_races['position']<=3).mean()*100:.2f}%")

# Build weather-specific models
print("\n" + "="*80)
print("BUILDING PREDICTIVE MODELS BY WEATHER CONDITION")
print("="*80)

weather_models = {}
weather_stats = {}

for weather in ['Sunny', 'Wet', 'Unknown']:
    weather_df = df[df['weather'] == weather].copy()

    X = weather_df[['aggressiveness', 'defensive_ability', 'stamina', 'laps']].copy()
    y = weather_df['time'].copy()

    model = LinearRegression()
    model.fit(X, y)

    weather_models[weather] = model
    weather_stats[weather] = {
        'count': len(weather_df),
        'proportion': len(weather_df) / len(df),
        'avg_laps': weather_df['laps'].mean()
    }

    print(f"\n{weather} Conditions:")
    print(f"  Races: {weather_stats[weather]['count']} ({weather_stats[weather]['proportion']*100:.1f}%)")
    print(f"  Avg Laps: {weather_stats[weather]['avg_laps']:.1f}")
    print(f"  Coefficients:")
    print(f"    Aggressiveness:    {model.coef_[0]:10.2f}")
    print(f"    Defensive Ability: {model.coef_[1]:10.2f}")
    print(f"    Stamina:           {model.coef_[2]:10.2f}")
    print(f"    Laps:              {model.coef_[3]:10.2f}")

# Calculate average laps across all races
avg_laps = df['laps'].mean()
print(f"\nOverall Average Laps: {avg_laps:.1f}")

# Define objective function (minimize expected race time)
def predict_time(stats, weather, avg_laps=avg_laps):
    """Predict race time given stats and weather condition"""
    agg, defense, stamina = stats
    model = weather_models[weather]
    X = np.array([[agg, defense, stamina, avg_laps]])
    return model.predict(X)[0]

def weighted_expected_time(stats):
    """Calculate weighted average race time across all weather conditions"""
    agg, defense, stamina = stats

    # Weight by weather frequency
    total_time = 0
    for weather in ['Sunny', 'Wet', 'Unknown']:
        time = predict_time([agg, defense, stamina], weather)
        weight = weather_stats[weather]['proportion']
        total_time += time * weight

    return total_time

def worst_case_time(stats):
    """Return worst-case time across all conditions"""
    times = [predict_time(stats, w) for w in ['Sunny', 'Wet', 'Unknown']]
    return max(times)

# Optimization with constraint that sum must be reasonable (not necessarily = 1)
print("\n" + "="*80)
print("OPTIMIZATION ANALYSIS")
print("="*80)

print("\nTesting different optimization strategies...")

# Strategy 1: Minimize expected time (weighted average)
print("\n" + "-"*80)
print("STRATEGY 1: Minimize Weighted Average Time")
print("-"*80)

def objective_weighted(x):
    return weighted_expected_time(x)

# Constraints: each stat between 0 and 1
bounds = [(0, 1), (0, 1), (0, 1)]

# Try multiple starting points to avoid local minima
best_result = None
best_score = float('inf')

for _ in range(20):
    x0 = np.random.random(3)
    result = minimize(objective_weighted, x0, bounds=bounds, method='L-BFGS-B')
    if result.fun < best_score:
        best_score = result.fun
        best_result = result

opt_stats_weighted = best_result.x
expected_time_weighted = weighted_expected_time(opt_stats_weighted)

print(f"\nOptimal Stats (Weighted Average):")
print(f"  Aggressiveness:    {opt_stats_weighted[0]:.4f}")
print(f"  Defensive Ability: {opt_stats_weighted[1]:.4f}")
print(f"  Stamina:           {opt_stats_weighted[2]:.4f}")
print(f"  Sum of stats:      {sum(opt_stats_weighted):.4f}")

print(f"\nExpected Performance:")
print(f"  Weighted Avg Time: {expected_time_weighted:.2f}s")
for weather in ['Sunny', 'Wet', 'Unknown']:
    time = predict_time(opt_stats_weighted, weather)
    print(f"    {weather:8s}: {time:.2f}s")

# Strategy 2: Minimize worst-case scenario (robust optimization)
print("\n" + "-"*80)
print("STRATEGY 2: Minimize Worst-Case Time (Robust)")
print("-"*80)

def objective_robust(x):
    return worst_case_time(x)

best_result_robust = None
best_score_robust = float('inf')

for _ in range(20):
    x0 = np.random.random(3)
    result = minimize(objective_robust, x0, bounds=bounds, method='L-BFGS-B')
    if result.fun < best_score_robust:
        best_score_robust = result.fun
        best_result_robust = result

opt_stats_robust = best_result_robust.x
worst_time_robust = worst_case_time(opt_stats_robust)

print(f"\nOptimal Stats (Robust):")
print(f"  Aggressiveness:    {opt_stats_robust[0]:.4f}")
print(f"  Defensive Ability: {opt_stats_robust[1]:.4f}")
print(f"  Stamina:           {opt_stats_robust[2]:.4f}")
print(f"  Sum of stats:      {sum(opt_stats_robust):.4f}")

print(f"\nExpected Performance:")
print(f"  Worst-Case Time: {worst_time_robust:.2f}s")
print(f"  Weighted Avg Time: {weighted_expected_time(opt_stats_robust):.2f}s")
for weather in ['Sunny', 'Wet', 'Unknown']:
    time = predict_time(opt_stats_robust, weather)
    print(f"    {weather:8s}: {time:.2f}s")

# Strategy 3: Optimize for winning (minimize time with variance consideration)
print("\n" + "-"*80)
print("STRATEGY 3: Balanced Performance Across Conditions")
print("-"*80)

def objective_balanced(x):
    times = [predict_time(x, w) for w in ['Sunny', 'Wet', 'Unknown']]
    # Minimize both average and variance (consistency)
    avg = np.mean(times)
    std = np.std(times)
    return avg + 0.1 * std  # Penalize variance

best_result_balanced = None
best_score_balanced = float('inf')

for _ in range(20):
    x0 = np.random.random(3)
    result = minimize(objective_balanced, x0, bounds=bounds, method='L-BFGS-B')
    if result.fun < best_score_balanced:
        best_score_balanced = result.fun
        best_result_balanced = result

opt_stats_balanced = best_result_balanced.x
times_balanced = [predict_time(opt_stats_balanced, w) for w in ['Sunny', 'Wet', 'Unknown']]

print(f"\nOptimal Stats (Balanced):")
print(f"  Aggressiveness:    {opt_stats_balanced[0]:.4f}")
print(f"  Defensive Ability: {opt_stats_balanced[1]:.4f}")
print(f"  Stamina:           {opt_stats_balanced[2]:.4f}")
print(f"  Sum of stats:      {sum(opt_stats_balanced):.4f}")

print(f"\nExpected Performance:")
print(f"  Weighted Avg Time: {weighted_expected_time(opt_stats_balanced):.2f}s")
print(f"  Std Deviation:     {np.std(times_balanced):.2f}s")
for weather in ['Sunny', 'Wet', 'Unknown']:
    time = predict_time(opt_stats_balanced, weather)
    print(f"    {weather:8s}: {time:.2f}s")

# Compare with current Tapped In Team stats
print("\n" + "="*80)
print("COMPARISON WITH CURRENT TAPPED IN TEAM")
print("="*80)

current_stats = [
    tapped_in_drivers['aggressiveness'].mean(),
    tapped_in_drivers['defensive_ability'].mean(),
    tapped_in_drivers['stamina'].mean()
]

current_expected = weighted_expected_time(current_stats)
current_times = [predict_time(current_stats, w) for w in ['Sunny', 'Wet', 'Unknown']]

print(f"\nCurrent Team Stats:")
print(f"  Aggressiveness:    {current_stats[0]:.4f}")
print(f"  Defensive Ability: {current_stats[1]:.4f}")
print(f"  Stamina:           {current_stats[2]:.4f}")

print(f"\nCurrent Predicted Performance:")
print(f"  Weighted Avg Time: {current_expected:.2f}s")
for i, weather in enumerate(['Sunny', 'Wet', 'Unknown']):
    print(f"    {weather:8s}: {current_times[i]:.2f}s")

print(f"\n" + "-"*80)
print("IMPROVEMENT POTENTIAL")
print("-"*80)

improvements = {
    'Weighted': (current_expected - expected_time_weighted, opt_stats_weighted),
    'Robust': (current_expected - weighted_expected_time(opt_stats_robust), opt_stats_robust),
    'Balanced': (current_expected - weighted_expected_time(opt_stats_balanced), opt_stats_balanced)
}

for strategy, (improvement, stats) in improvements.items():
    print(f"\n{strategy} Strategy:")
    print(f"  Time Improvement:  {improvement:.2f}s ({improvement/current_expected*100:.2f}%)")

# Analyze top performers in dataset
print("\n" + "="*80)
print("VALIDATION: ANALYZING TOP PERFORMING DRIVERS")
print("="*80)

# Get drivers with best average times (min 50 races)
driver_performance = df.groupby('driver').agg({
    'time': ['mean', 'count'],
    'position': 'mean'
}).round(2)
driver_performance.columns = ['avg_time', 'races', 'avg_position']
driver_performance = driver_performance[driver_performance['races'] >= 50]

top_performers = driver_performance.nsmallest(5, 'avg_time')
print("\nTop 5 Drivers by Average Time:")
print(top_performers)

print("\nTop Performers' Attributes:")
for driver in top_performers.index:
    driver_info = drivers[drivers['driver_name'] == driver].iloc[0]
    print(f"\n{driver}:")
    print(f"  Aggressiveness:    {driver_info['aggressiveness']:.3f}")
    print(f"  Defensive Ability: {driver_info['defensive_ability']:.3f}")
    print(f"  Stamina:           {driver_info['stamina']:.3f}")

top_performer_stats = drivers[drivers['driver_name'].isin(top_performers.index)]
print(f"\nTop Performers Average Stats:")
print(f"  Aggressiveness:    {top_performer_stats['aggressiveness'].mean():.4f}")
print(f"  Defensive Ability: {top_performer_stats['defensive_ability'].mean():.4f}")
print(f"  Stamina:           {top_performer_stats['stamina'].mean():.4f}")

# Final recommendation
print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

# Choose the strategy that best balances performance
final_stats = opt_stats_weighted  # Weighted average typically best for overall winning

print(f"\nRECOMMENDED OPTIMAL STATS FOR TAPPED IN TEAM:")
print(f"\n  🎯 Aggressiveness:    {final_stats[0]:.3f}")
print(f"  🛡️  Defensive Ability: {final_stats[1]:.3f}")
print(f"  💪 Stamina:           {final_stats[2]:.3f}")

print(f"\nExpected Performance Improvement:")
print(f"  Current Avg Time:  {current_expected:.2f}s")
print(f"  Optimized Avg Time: {expected_time_weighted:.2f}s")
print(f"  Improvement:       {current_expected - expected_time_weighted:.2f}s ({(current_expected - expected_time_weighted)/current_expected*100:.2f}%)")

print(f"\nPerformance by Weather:")
for weather in ['Sunny', 'Wet', 'Unknown']:
    current = predict_time(current_stats, weather)
    optimized = predict_time(final_stats, weather)
    improvement = current - optimized
    print(f"  {weather:8s}: {current:.0f}s → {optimized:.0f}s (Δ{improvement:+.0f}s)")

print(f"\n" + "-"*80)
print("KEY INSIGHTS")
print("-"*80)

print(f"\n1. AGGRESSIVENESS should be: {final_stats[0]:.3f}")
if final_stats[0] < 0.3:
    print("   → VERY LOW (defensive, controlled driving wins)")
elif final_stats[0] < 0.5:
    print("   → LOW to MODERATE (balanced approach)")
else:
    print("   → HIGH (aggressive racing)")

print(f"\n2. DEFENSIVE ABILITY should be: {final_stats[1]:.3f}")
if final_stats[1] > 0.7:
    print("   → VERY HIGH (critical for winning)")
elif final_stats[1] > 0.5:
    print("   → HIGH (important skill)")
else:
    print("   → MODERATE (balanced)")

print(f"\n3. STAMINA should be: {final_stats[2]:.3f}")
if final_stats[2] < 0.3:
    print("   → LOW (counterintuitively better in most conditions)")
elif final_stats[2] < 0.5:
    print("   → MODERATE")
else:
    print("   → HIGH (helps in long races)")

print(f"\n4. This allocation is optimized for:")
print(f"   ✓ Weighted performance across all weather conditions")
print(f"   ✓ {weather_stats['Sunny']['proportion']*100:.0f}% Sunny races (slowest conditions)")
print(f"   ✓ {weather_stats['Wet']['proportion']*100:.0f}% Wet races (fastest conditions)")
print(f"   ✓ {weather_stats['Unknown']['proportion']*100:.0f}% Unknown races (skill-based)")

print("\n" + "="*80)

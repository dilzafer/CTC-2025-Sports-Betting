import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

# Load data
races = pd.read_csv('all_race_results.csv')
drivers = pd.read_csv('driver_roster.csv')

# Merge datasets
df = races.merge(drivers[['driver_name', 'aggressiveness', 'defensive_ability', 'stamina', 'age']],
                 left_on='driver', right_on='driver_name', how='left')
df = df.drop('driver_name', axis=1)

print("="*80)
print("WEATHER-SPECIFIC REGRESSION ANALYSIS")
print("="*80)

# Split by weather condition
weather_conditions = ['Sunny', 'Wet', 'Unknown']
feature_cols = ['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']

results_summary = []

for weather in weather_conditions:
    print(f"\n{'='*80}")
    print(f"WEATHER CONDITION: {weather.upper()}")
    print(f"{'='*80}")

    # Filter data for this weather condition
    weather_df = df[df['weather'] == weather].copy()

    print(f"\nSample size: {len(weather_df)} races")
    print(f"Average time: {weather_df['time'].mean():.2f}s")
    print(f"Std deviation: {weather_df['time'].std():.2f}s")

    # Correlation analysis
    print(f"\n{'-'*80}")
    print(f"CORRELATIONS WITH RACE TIME ({weather} conditions)")
    print(f"{'-'*80}")

    correlations = weather_df[['time', 'aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].corr()['time'].sort_values(ascending=False)
    print("\nPearson Correlations:")
    for feature, corr in correlations.items():
        if feature != 'time':
            # Calculate p-value
            _, p_val = stats.pearsonr(weather_df[feature], weather_df['time'])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {feature:25s}: {corr:7.4f}  (p={p_val:.4f}) {sig}")

    # Regression model for this weather
    X = weather_df[feature_cols].copy()
    y = weather_df['time'].copy()

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)

    print(f"\n{'-'*80}")
    print(f"REGRESSION MODEL PERFORMANCE ({weather})")
    print(f"{'-'*80}")
    print(f"R² Score: {r2:.4f} (explains {r2*100:.2f}% of variance)")

    # Feature importance
    print(f"\nRegression Coefficients (Impact on Race Time):")
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)

    for idx, row in coef_df.iterrows():
        direction = "SLOWER" if row['Coefficient'] > 0 else "FASTER"
        print(f"  {row['Feature']:25s}: {row['Coefficient']:10.2f}  ({direction})")

    print(f"\n  Intercept: {model.intercept_:.2f}")

    # Store results for comparison
    results_summary.append({
        'Weather': weather,
        'Sample_Size': len(weather_df),
        'Avg_Time': weather_df['time'].mean(),
        'R2': r2,
        'Top_Factor': coef_df.iloc[0]['Feature'],
        'Top_Coefficient': coef_df.iloc[0]['Coefficient'],
        'Aggressiveness_Coef': coef_df[coef_df['Feature'] == 'aggressiveness']['Coefficient'].values[0],
        'Defensive_Coef': coef_df[coef_df['Feature'] == 'defensive_ability']['Coefficient'].values[0],
        'Stamina_Coef': coef_df[coef_df['Feature'] == 'stamina']['Coefficient'].values[0],
        'Laps_Coef': coef_df[coef_df['Feature'] == 'laps']['Coefficient'].values[0]
    })

    # Winner characteristics in this weather
    print(f"\n{'-'*80}")
    print(f"WINNING CHARACTERISTICS ({weather} conditions)")
    print(f"{'-'*80}")

    winners = weather_df[weather_df['position'] == 1]
    print(f"\nAverage attributes of race WINNERS (n={len(winners)}):")
    print(f"  Aggressiveness:    {winners['aggressiveness'].mean():.3f} (std: {winners['aggressiveness'].std():.3f})")
    print(f"  Defensive Ability: {winners['defensive_ability'].mean():.3f} (std: {winners['defensive_ability'].std():.3f})")
    print(f"  Stamina:           {winners['stamina'].mean():.3f} (std: {winners['stamina'].std():.3f})")
    print(f"  Age:               {winners['age'].mean():.1f} (std: {winners['age'].std():.1f})")

    losers = weather_df[weather_df['position'] >= 15]
    print(f"\nAverage attributes of POOR PERFORMERS (position ≥15, n={len(losers)}):")
    print(f"  Aggressiveness:    {losers['aggressiveness'].mean():.3f} (std: {losers['aggressiveness'].std():.3f})")
    print(f"  Defensive Ability: {losers['defensive_ability'].mean():.3f} (std: {losers['defensive_ability'].std():.3f})")
    print(f"  Stamina:           {losers['stamina'].mean():.3f} (std: {losers['stamina'].std():.3f})")
    print(f"  Age:               {losers['age'].mean():.1f} (std: {losers['age'].std():.1f})")

    # Statistical comparison
    print(f"\n{'-'*80}")
    print(f"WINNER vs POOR PERFORMER COMPARISON ({weather})")
    print(f"{'-'*80}")

    for attr in ['aggressiveness', 'defensive_ability', 'stamina', 'age']:
        t_stat, p_val = stats.ttest_ind(winners[attr], losers[attr])
        diff = winners[attr].mean() - losers[attr].mean()
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {attr:25s}: diff={diff:+.3f}, p={p_val:.4f} {sig}")

# Compare across weather conditions
print(f"\n\n{'='*80}")
print(f"CROSS-WEATHER COMPARISON")
print(f"{'='*80}")

comparison_df = pd.DataFrame(results_summary)
print("\nModel Performance by Weather:")
print(comparison_df[['Weather', 'Sample_Size', 'Avg_Time', 'R2']].to_string(index=False))

print(f"\n{'-'*80}")
print("COEFFICIENT COMPARISON: Impact of Each Attribute by Weather")
print(f"{'-'*80}")

print("\nAggressiveness Coefficient (higher = more penalty):")
for _, row in comparison_df.iterrows():
    print(f"  {row['Weather']:10s}: {row['Aggressiveness_Coef']:10.2f}")

print("\nDefensive Ability Coefficient (more negative = more beneficial):")
for _, row in comparison_df.iterrows():
    print(f"  {row['Weather']:10s}: {row['Defensive_Coef']:10.2f}")

print("\nStamina Coefficient:")
for _, row in comparison_df.iterrows():
    print(f"  {row['Weather']:10s}: {row['Stamina_Coef']:10.2f}")

print("\nLaps Coefficient (difficulty multiplier):")
for _, row in comparison_df.iterrows():
    print(f"  {row['Weather']:10s}: {row['Laps_Coef']:10.2f}")

# Key insights
print(f"\n\n{'='*80}")
print(f"KEY INSIGHTS & RECOMMENDATIONS")
print(f"{'='*80}")

print("\n1. MOST PREDICTIVE WEATHER CONDITION:")
best_r2 = comparison_df.loc[comparison_df['R2'].idxmax()]
print(f"   {best_r2['Weather']} (R² = {best_r2['R2']:.4f})")
print(f"   → Driver attributes matter MOST in {best_r2['Weather']} conditions")

print("\n2. AGGRESSIVENESS PENALTY BY WEATHER:")
agg_penalties = comparison_df.sort_values('Aggressiveness_Coef', ascending=False)
print(f"   Worst penalty: {agg_penalties.iloc[0]['Weather']} (+{agg_penalties.iloc[0]['Aggressiveness_Coef']:.0f}s per unit)")
print(f"   Best penalty:  {agg_penalties.iloc[-1]['Weather']} (+{agg_penalties.iloc[-1]['Aggressiveness_Coef']:.0f}s per unit)")
print(f"   → Aggressive driving hurts MOST in {agg_penalties.iloc[0]['Weather']} weather")

print("\n3. DEFENSIVE ABILITY ADVANTAGE BY WEATHER:")
def_benefits = comparison_df.sort_values('Defensive_Coef', ascending=True)
print(f"   Best advantage:  {def_benefits.iloc[0]['Weather']} ({def_benefits.iloc[0]['Defensive_Coef']:.0f}s per unit)")
print(f"   Least advantage: {def_benefits.iloc[-1]['Weather']} ({def_benefits.iloc[-1]['Defensive_Coef']:.0f}s per unit)")
print(f"   → Defensive skills matter MOST in {def_benefits.iloc[0]['Weather']} conditions")

print("\n4. STAMINA IMPACT BY WEATHER:")
stamina_impact = comparison_df.sort_values('Stamina_Coef', ascending=False)
print(f"   Highest impact: {stamina_impact.iloc[0]['Weather']} ({stamina_impact.iloc[0]['Stamina_Coef']:.0f}s per unit)")
print(f"   Lowest impact:  {stamina_impact.iloc[-1]['Weather']} ({stamina_impact.iloc[-1]['Stamina_Coef']:.0f}s per unit)")

print("\n5. RACE LENGTH DIFFICULTY BY WEATHER:")
laps_difficulty = comparison_df.sort_values('Laps_Coef', ascending=False)
print(f"   Hardest: {laps_difficulty.iloc[0]['Weather']} (+{laps_difficulty.iloc[0]['Laps_Coef']:.2f}s per lap)")
print(f"   Easiest: {laps_difficulty.iloc[-1]['Weather']} (+{laps_difficulty.iloc[-1]['Laps_Coef']:.2f}s per lap)")
print(f"   → Long races are hardest in {laps_difficulty.iloc[0]['Weather']} weather")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

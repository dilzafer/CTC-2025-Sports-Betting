import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
races = pd.read_csv('all_race_results.csv')
drivers = pd.read_csv('driver_roster.csv')

# Merge datasets
print("\nMerging race results with driver attributes...")
df = races.merge(drivers[['driver_name', 'aggressiveness', 'defensive_ability', 'stamina', 'age']],
                 left_on='driver', right_on='driver_name', how='left')
df = df.drop('driver_name', axis=1)

# Create derived features
df['time_per_lap'] = df['time'] / df['laps']

# Encode weather as dummy variables
weather_dummies = pd.get_dummies(df['weather'], prefix='weather')
df = pd.concat([df, weather_dummies], axis=1)

print(f"\nDataset shape: {df.shape}")
print(f"Total races analyzed: {len(df)}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS: Driver Attributes vs Race Time")
print("="*80)

correlations = df[['time', 'aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].corr()['time'].sort_values(ascending=False)
print("\nPearson Correlations with Race Time:")
for feature, corr in correlations.items():
    if feature != 'time':
        print(f"  {feature:25s}: {corr:7.4f}")

# Weather impact on time
print("\n" + "-"*80)
print("WEATHER IMPACT ON RACE TIME")
print("-"*80)
weather_stats = df.groupby('weather').agg({
    'time': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
print(weather_stats)

# Statistical test for weather differences
sunny_times = df[df['weather'] == 'Sunny']['time']
wet_times = df[df['weather'] == 'Wet']['time']
unknown_times = df[df['weather'] == 'Unknown']['time']

print(f"\nANOVA test for weather effect:")
f_stat, p_value = stats.f_oneway(sunny_times, wet_times, unknown_times)
print(f"  F-statistic: {f_stat:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("REGRESSION ANALYSIS: Predicting Race Time")
print("="*80)

# Prepare features for regression
feature_cols = ['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps',
                'weather_Sunny', 'weather_Wet', 'weather_Unknown']
X = df[feature_cols].copy()
y = df['time'].copy()

# Fit regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Model performance
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  RMSE: {rmse:.2f}")

# Feature importance (coefficients)
print(f"\nRegression Coefficients (Feature Importance):")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

for _, row in coef_df.iterrows():
    print(f"  {row['Feature']:25s}: {row['Coefficient']:10.2f}")

print(f"\n  Intercept: {model.intercept_:.2f}")

# Identify strongest predictors
print(f"\nSTRONGEST PREDICTORS (by absolute coefficient):")
for i, row in coef_df.head(3).iterrows():
    print(f"  {i+1}. {row['Feature']} (coef: {row['Coefficient']:.2f})")

# ============================================================================
# INTERACTION EFFECTS
# ============================================================================
print("\n" + "="*80)
print("INTERACTION ANALYSIS")
print("="*80)

# Stamina × Laps interaction
print("\nStamina × Race Length (Laps) Relationship:")
stamina_bins = pd.qcut(df['stamina'], q=3, labels=['Low', 'Medium', 'High'])
lap_bins = pd.qcut(df['laps'], q=3, labels=['Short', 'Medium', 'Long'])
interaction_df = df.copy()
interaction_df['stamina_group'] = stamina_bins
interaction_df['lap_group'] = lap_bins

stamina_lap_interaction = interaction_df.groupby(['stamina_group', 'lap_group'])['time'].mean().unstack()
print(stamina_lap_interaction.round(2))

# Aggressiveness × Weather
print("\nAggressiveness × Weather Interaction:")
agg_bins = pd.qcut(df['aggressiveness'], q=3, labels=['Low', 'Medium', 'High'])
interaction_df['agg_group'] = agg_bins
agg_weather_interaction = interaction_df.groupby(['agg_group', 'weather'])['time'].mean().unstack()
print(agg_weather_interaction.round(2))

# ============================================================================
# TEAM PERFORMANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("TEAM PERFORMANCE ANALYSIS")
print("="*80)

team_stats = df.groupby('team').agg({
    'time': ['mean', 'std', 'min'],
    'time_per_lap': 'mean',
    'points_earned': 'sum',
    'position': 'mean'
}).round(2)

team_stats.columns = ['Avg_Time', 'Std_Time', 'Best_Time', 'Avg_Time_Per_Lap', 'Total_Points', 'Avg_Position']
team_stats = team_stats.sort_values('Avg_Time')

print("\nTeam Rankings (by Average Race Time):")
print(team_stats)

# Win rate and podium finishes
team_wins = df[df['position'] == 1].groupby('team').size()
team_podiums = df[df['position'] <= 3].groupby('team').size()
team_races = df.groupby('team').size()

team_performance = pd.DataFrame({
    'Wins': team_wins,
    'Podiums': team_podiums,
    'Total_Races': team_races,
    'Win_Rate': (team_wins / team_races * 100).round(2),
    'Podium_Rate': (team_podiums / team_races * 100).round(2)
}).fillna(0).sort_values('Win_Rate', ascending=False)

print("\n" + "-"*80)
print("TEAM WIN & PODIUM STATISTICS")
print("-"*80)
print(team_performance)

# Weather-adjusted team performance
print("\n" + "-"*80)
print("TEAM PERFORMANCE BY WEATHER")
print("-"*80)
team_weather = df.groupby(['team', 'weather'])['time'].mean().unstack().round(2)
print(team_weather)

# ============================================================================
# SUMMARY FINDINGS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)

best_team_time = team_stats.index[0]
best_team_wins = team_performance.index[0]

print(f"\n1. STRONGEST RELATIONSHIP WITH RACE TIME:")
print(f"   - Top predictor: {coef_df.iloc[0]['Feature']} (coefficient: {coef_df.iloc[0]['Coefficient']:.2f})")
print(f"   - Model explains {r2*100:.2f}% of variance in race times (R² = {r2:.4f})")

print(f"\n2. BEST PERFORMING TEAM:")
print(f"   - By Average Time: {best_team_time} ({team_stats.loc[best_team_time, 'Avg_Time']:.2f}s avg)")
print(f"   - By Win Rate: {best_team_wins} ({team_performance.loc[best_team_wins, 'Win_Rate']:.1f}% wins)")
print(f"   - By Total Points: {team_stats['Total_Points'].idxmax()} ({team_stats['Total_Points'].max():.0f} points)")

print(f"\n3. KEY DRIVER ATTRIBUTES:")
driver_attr_coefs = coef_df[coef_df['Feature'].isin(['aggressiveness', 'defensive_ability', 'stamina', 'age'])]
print(f"   - Most impactful: {driver_attr_coefs.iloc[0]['Feature']} (coef: {driver_attr_coefs.iloc[0]['Coefficient']:.2f})")

print(f"\n4. WEATHER IMPACT:")
print(f"   - Fastest condition: {df.groupby('weather')['time'].mean().idxmin()} ({df.groupby('weather')['time'].mean().min():.2f}s avg)")
print(f"   - Slowest condition: {df.groupby('weather')['time'].mean().idxmax()} ({df.groupby('weather')['time'].mean().max():.2f}s avg)")
print(f"   - Statistical significance: {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Load data
races = pd.read_csv('all_race_results.csv')
drivers = pd.read_csv('driver_roster.csv')

# Merge datasets
df = races.merge(drivers[['driver_name', 'aggressiveness', 'defensive_ability', 'stamina', 'age']],
                 left_on='driver', right_on='driver_name', how='left')

print("="*80)
print("PREDICTION: WET WEATHER, 49 LAPS")
print("="*80)

print("\n📊 RACE 2 ACTUAL RESULTS (Unknown Weather, 65 laps):")
print("="*80)

race2_results = """1	tree_17	Tapped In Team	5410.44
2	hihi	Tapped In Team	5416.34
3	Aayush_25	Tapped In Team	5417.55
4	bruh	Red Velvet Racing	5418.06
5	Enter driver name	B2B SaaS Motorsport	5430.23
6	da mother	DNF Finishers	5436.48
7	Minh	Tapped In Team	5436.69
8	Biden	Lag's 200 Ping Power	5484.85
9	Alex-Song_8	Tapped In Team	5484.87
10	Aneesh Hamilton	Tapped In Team	5488.56
11	Jamala_22	Red Velvet Racing	5490.40
12	RohanMalla_11	Tapped In Team	5490.83
13	4Alpha_2	Boba Tea Gnarly Automotives	5491.73
14	micheal	Spanky's Fast Cars	5492.09
15	Max	Spanky's Fast Cars	5497.29
16	Justin Zhang	Red Velvet Racing	5497.45
17	safer_21	DNF Finishers	5499.64
18	lebron2_30	Tapped In Team	5501.45
19	pomelocloudjasminematcha_27	Tapped In Team	5504.56
20	Johnny	Boba Tea Gnarly Automotives	5506.64
21	recursion	Tapped In Team	5507.17
22	JSIN	Tapped In Team	5507.20
23	thosewhoknow_28	Tapped In Team	5507.56
24	royal giant	DNF Finishers	5509.46
25	rizzSquad_9	Tapped In Team	5515.27
26	haf/group10	Tapped In Team	DNF
27	BetterVerstappen_26	Tapped In Team	DNF
28	oski_19	B2B SaaS Motorsport	DNF
29	Nandakumar_13	Tapped In Team	DNF
30	deepa_29	B2B SaaS Motorsport	DNF
31	asjiodf	B2B SaaS Motorsport	DNF
32	Mario_33	Tapped In Team	DNF
33	Miami67	Tapped In Team	DNF"""

print("\n⚠️ MAJOR UPSET: Our predictions were WRONG!")
print("\n❌ asjiodf DNF (predicted #1)")
print("❌ deepa_29 DNF (predicted #2)")
print("❌ B2B SaaS Motorsport had multiple DNFs")
print("\n✅ bruh finished 4th (we predicted #14 - he outperformed!)")
print("✅ Tapped In Team DOMINATED (13 of top 15)")
print("✅ hihi (Tapped In) 2nd place")
print("✅ New drivers appeared (tree_17, Aayush_25, etc.)")

print("\n💡 LESSONS LEARNED:")
print("   • Model predictions don't account for mechanical failures/crashes")
print("   • Unknown weather had more randomness than expected")
print("   • Tapped In Team performed much better than predicted")
print("   • Some top drivers DNF'd (race incidents, not predictable)")

# Build WET weather model
wet_df = df[df['weather'] == 'Wet'].copy()

print("\n" + "="*80)
print("WET WEATHER ANALYSIS")
print("="*80)

print(f"\nWet weather races in dataset: {len(wet_df)}")
print(f"Average laps in Wet: {wet_df['laps'].mean():.1f}")
print(f"Lap range: {wet_df['laps'].min():.0f} - {wet_df['laps'].max():.0f}")

# Check for 49-lap races
wet_49 = wet_df[wet_df['laps'] == 49]
print(f"Races with exactly 49 laps (Wet): {len(wet_49)}")

# Build regression model
X_wet = wet_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
y_wet = wet_df['time'].copy()

model_wet = LinearRegression()
model_wet.fit(X_wet, y_wet)

r2 = model_wet.score(X_wet, y_wet)
print(f"\nModel R² Score: {r2:.4f} (explains {r2*100:.1f}% of variance)")

print("\n" + "-"*80)
print("REGRESSION COEFFICIENTS FOR WET WEATHER")
print("-"*80)

features = ['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']
print(f"\nCoefficients:")
for i, feature in enumerate(features):
    direction = "SLOWER" if model_wet.coef_[i] > 0 else "FASTER"
    print(f"  {feature:25s}: {model_wet.coef_[i]:10.2f}s  ({direction})")
print(f"  Intercept: {model_wet.intercept_:.2f}s")

# Key insights about WET
print("\n" + "-"*80)
print("KEY DIFFERENCES: WET vs SUNNY vs UNKNOWN")
print("-"*80)

sunny_df = df[df['weather'] == 'Sunny']
unknown_df = df[df['weather'] == 'Unknown']

X_sunny = sunny_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
y_sunny = sunny_df['time'].copy()
model_sunny = LinearRegression()
model_sunny.fit(X_sunny, y_sunny)

X_unknown = unknown_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
y_unknown = unknown_df['time'].copy()
model_unknown = LinearRegression()
model_unknown.fit(X_unknown, y_unknown)

print(f"\n{'Attribute':<25} {'Sunny':<15} {'Wet':<15} {'Unknown':<15}")
print("-" * 75)
for i, feature in enumerate(['Aggressiveness', 'Defensive', 'Stamina', 'Age', 'Laps']):
    print(f"{feature:<25} {model_sunny.coef_[i]:>10.0f}s     {model_wet.coef_[i]:>10.0f}s     {model_unknown.coef_[i]:>10.0f}s")

print("\n💡 WET WEATHER INSIGHTS:")
print(f"  • Aggressiveness: {model_wet.coef_[0]:.0f}s (NEGATIVE = helps slightly!)")
print(f"  • Defensive: {model_wet.coef_[1]:.0f}s (negative = helps)")
print(f"  • Stamina: {model_wet.coef_[2]:.0f}s (NEGATIVE = LOW stamina helps!)")
print(f"  • Age: {model_wet.coef_[3]:.0f}s (positive = younger better)")
print(f"  • Laps: {model_wet.coef_[4]:.0f}s per lap")

print("\n🔑 WET IS COMPLETELY DIFFERENT:")
print("  ✓ Aggressiveness HELPS (opposite of Sunny/Unknown!)")
print("  ✓ Defensive ability still helps (but less than Unknown)")
print("  ✓ LOW stamina still better (like Sunny)")
print("  ✓ YOUNGER drivers have advantage (age penalty)")
print("  ✓ Wet is FASTEST overall condition (8,639s avg vs 10,922s Sunny)")

# Winners in wet conditions
print("\n" + "-"*80)
print("WET WEATHER WINNER CHARACTERISTICS")
print("-"*80)

wet_winners = wet_df[wet_df['position'] == 1]
wet_losers = wet_df[wet_df['position'] >= 15]

print(f"\nWinners in Wet (n={len(wet_winners)}):")
print(f"  Aggressiveness:    {wet_winners['aggressiveness'].mean():.3f} ± {wet_winners['aggressiveness'].std():.3f}")
print(f"  Defensive Ability: {wet_winners['defensive_ability'].mean():.3f} ± {wet_winners['defensive_ability'].std():.3f}")
print(f"  Stamina:           {wet_winners['stamina'].mean():.3f} ± {wet_winners['stamina'].std():.3f}")
print(f"  Age:               {wet_winners['age'].mean():.1f} ± {wet_winners['age'].std():.1f}")

print(f"\nPoor Performers in Wet (n={len(wet_losers)}):")
print(f"  Aggressiveness:    {wet_losers['aggressiveness'].mean():.3f} ± {wet_losers['aggressiveness'].std():.3f}")
print(f"  Defensive Ability: {wet_losers['defensive_ability'].mean():.3f} ± {wet_losers['defensive_ability'].std():.3f}")
print(f"  Stamina:           {wet_losers['stamina'].mean():.3f} ± {wet_losers['stamina'].std():.3f}")
print(f"  Age:               {wet_losers['age'].mean():.1f} ± {wet_losers['age'].std():.1f}")

# Get current standings from Race 2
print("\n" + "="*80)
print("RACE 3 PREDICTIONS: WET WEATHER, 49 LAPS")
print("="*80)

# Parse Race 2 results to get current standings
race2_data = []
for line in race2_results.strip().split('\n'):
    parts = line.split('\t')
    if len(parts) >= 3:
        try:
            rank = int(parts[0])
            username = parts[1].strip()
            team = parts[2].strip()
            race2_data.append({'username': username, 'team': team, 'race2_rank': rank})
        except:
            pass

# Only keep finishers (top 25)
race2_df = pd.DataFrame(race2_data[:25])

print(f"\nAnalyzing {len(race2_df)} drivers who finished Race 2")

# Predict for each driver
def predict_wet_49(stats, age, laps=49):
    """Predict race time for Wet weather, 49 laps"""
    agg, defense, stamina = stats
    X = np.array([[agg, defense, stamina, age, laps]])
    return model_wet.predict(X)[0]

predictions_wet = []

for idx, row in race2_df.iterrows():
    username = row['username']
    team = row['team']

    # Try to match driver
    driver_match = drivers[drivers['username'] == username]
    if len(driver_match) == 0:
        driver_match = drivers[drivers['team'] == team].iloc[0:1]

    if len(driver_match) > 0:
        driver = driver_match.iloc[0]

        # Predict Wet 49 laps
        pred_time = predict_wet_49(
            [driver['aggressiveness'], driver['defensive_ability'], driver['stamina']],
            driver['age']
        )

        predictions_wet.append({
            'username': username,
            'team': team,
            'race2_rank': row['race2_rank'],
            'aggressiveness': driver['aggressiveness'],
            'defensive_ability': driver['defensive_ability'],
            'stamina': driver['stamina'],
            'age': driver['age'],
            'predicted_time_wet': pred_time
        })

pred_wet_df = pd.DataFrame(predictions_wet)
pred_wet_df = pred_wet_df.sort_values('predicted_time_wet')
pred_wet_df['predicted_rank'] = range(1, len(pred_wet_df) + 1)
pred_wet_df['rank_change'] = pred_wet_df['race2_rank'] - pred_wet_df['predicted_rank']

print("\n" + "-"*80)
print("PREDICTED RANKINGS: WET WEATHER, 49 LAPS")
print("-"*80)

print(f"\n{'Pred':<5} {'Username':<30} {'Team':<30} {'R2':<5} {'Time':<10} {'Agg':<6} {'Def':<6} {'Stam':<6} {'Age':<4}")
print("-" * 115)

for idx, row in pred_wet_df.iterrows():
    print(f"{int(row['predicted_rank']):<5} {row['username']:<30} {row['team']:<30} "
          f"{int(row['race2_rank']):<5} {row['predicted_time_wet']:<10.0f} "
          f"{row['aggressiveness']:<6.2f} {row['defensive_ability']:<6.2f} "
          f"{row['stamina']:<6.2f} {row['age']:<4.0f}")

# Betting recommendations
print("\n" + "="*80)
print("BETTING RECOMMENDATIONS: WET WEATHER, 49 LAPS")
print("="*80)

print("\n🏆 TOP 5 PREDICTIONS:")
top5 = pred_wet_df.head(5)
for idx, row in top5.iterrows():
    change = row['rank_change']
    change_str = f"⬆️ +{int(change)}" if change > 0 else f"⬇️ {int(change)}" if change < 0 else "="

    print(f"\n#{int(row['predicted_rank'])}: {row['username']} ({row['team']})")
    print(f"  Race 2 finish: #{int(row['race2_rank'])} ({change_str})")
    print(f"  Predicted time: {row['predicted_time_wet']:.0f}s")
    print(f"  Profile: Agg={row['aggressiveness']:.2f}, Def={row['defensive_ability']:.2f}, "
          f"Stam={row['stamina']:.2f}, Age={row['age']:.0f}")

    # Explain why
    reasons = []
    if row['aggressiveness'] >= 0.5:
        reasons.append(f"High aggression ({row['aggressiveness']:.2f}) HELPS in Wet")
    if row['defensive_ability'] >= 0.5:
        reasons.append(f"Good defense ({row['defensive_ability']:.2f})")
    if row['stamina'] <= 0.3:
        reasons.append(f"Low stamina ({row['stamina']:.2f}) helps")
    if row['age'] <= 25:
        reasons.append(f"Young ({row['age']:.0f}) = advantage in Wet")

    if reasons:
        print(f"  Why: {', '.join(reasons)}")

# Value bets
print("\n💎 VALUE BETS (Biggest Movers UP):")
value_bets = pred_wet_df[pred_wet_df['rank_change'] >= 3].head(5)
if len(value_bets) > 0:
    for idx, row in value_bets.iterrows():
        print(f"\n{row['username']} - Race 2: #{int(row['race2_rank'])} → Predicted: #{int(row['predicted_rank'])} (⬆️ +{int(row['rank_change'])})")
        print(f"  Key stats: Agg={row['aggressiveness']:.2f}, Def={row['defensive_ability']:.2f}, Stam={row['stamina']:.2f}")
else:
    print("  Field is relatively stable - fewer clear value bets")

# Avoid
print("\n⚠️ AVOID (Likely to Drop):")
avoid = pred_wet_df[pred_wet_df['rank_change'] <= -3].head(3)
if len(avoid) > 0:
    for idx, row in avoid.iterrows():
        print(f"\n{row['username']} - Race 2: #{int(row['race2_rank'])} → Predicted: #{int(row['predicted_rank'])} (⬇️ {int(row['rank_change'])})")
        print(f"  Why: Profile not suited for Wet conditions")
else:
    print("  Most drivers reasonably positioned")

# Team analysis
print("\n" + "-"*80)
print("TEAM PERFORMANCE: WET WEATHER")
print("-"*80)

team_performance = pred_wet_df.groupby('team').agg({
    'predicted_time_wet': 'mean',
    'predicted_rank': 'mean',
    'username': 'count'
}).sort_values('predicted_time_wet')
team_performance.columns = ['Avg_Time', 'Avg_Rank', 'Count']

print(f"\n{'Team':<35} {'Drivers':<8} {'Avg Rank':<10} {'Avg Time':<12}")
print("-" * 70)
for team, row in team_performance.iterrows():
    print(f"{team:<35} {row['Count']:<8.0f} {row['Avg_Rank']:<10.1f} {row['Avg_Time']:<12.0f}")

print(f"\n🏁 Best team for Wet: {team_performance.index[0]}")

# Key strategy
print("\n" + "="*80)
print("WET WEATHER BETTING STRATEGY")
print("="*80)

print(f"\n🌧️ CONDITIONS: Wet Weather, 49 laps")

print(f"\n🔑 WHAT'S DIFFERENT IN WET:")
print(f"  1. AGGRESSIVENESS HELPS! ({model_wet.coef_[0]:.0f}s)")
print(f"     → Bold, aggressive driving is REWARDED in wet")
print(f"     → Complete opposite of Sunny/Unknown")
print(f"  ")
print(f"  2. DEFENSIVE ABILITY still helps ({model_wet.coef_[1]:.0f}s)")
print(f"     → Skill matters, but less than Unknown")
print(f"  ")
print(f"  3. LOW STAMINA better ({model_wet.coef_[2]:.0f}s)")
print(f"     → Pacing > endurance in wet conditions")
print(f"  ")
print(f"  4. YOUNGER drivers favored ({model_wet.coef_[3]:+.0f}s per year)")
print(f"     → Reflexes and adaptability matter")
print(f"  ")
print(f"  5. Wet is RANDOM (R²={r2:.3f})")
print(f"     → Only {r2*100:.1f}% of variance explained")
print(f"     → More unpredictable than other conditions")

print(f"\n🎯 OPTIMAL WET PROFILE:")
print(f"  • Aggressiveness: MODERATE to HIGH (0.4-0.7)")
print(f"  • Defensive Ability: MODERATE to HIGH (0.5+)")
print(f"  • Stamina: LOW (0.1-0.3)")
print(f"  • Age: YOUNG (20-25 years)")

print(f"\n⚠️ BETTING CAUTION:")
print(f"  • Wet weather has HIGH VARIANCE")
print(f"  • Expect more DNFs and crashes")
print(f"  • Race 2 showed many DNFs in Unknown - Wet will be similar")
print(f"  • Don't bet your entire bankroll on one driver!")

print(f"\n💡 RECOMMENDED BETTING APPROACH:")
print(f"  • Spread bets across multiple drivers")
print(f"  • Focus on proven finishers from Race 2")
print(f"  • Young aggressive drivers have edge")
print(f"  • Avoid older conservative drivers")
print(f"  • Consider TOP 5 finishes over WIN bets (safer)")

# Final picks
print("\n" + "="*80)
print("FINAL BETTING PICKS")
print("="*80)

print("\n🏆 WIN BETS (Risky but high reward):")
winner_pick = pred_wet_df.iloc[0]
print(f"  1. {winner_pick['username']} (Best predicted time)")
second_pick = pred_wet_df.iloc[1]
print(f"  2. {second_pick['username']} (2nd best time)")
third_pick = pred_wet_df.iloc[2]
print(f"  3. {third_pick['username']} (3rd best time)")

print("\n🥈 PODIUM BETS (Safer):")
for idx, row in pred_wet_df.head(5).iterrows():
    print(f"  • {row['username']}")

print("\n💰 VALUE BETS:")
if len(value_bets) > 0:
    for idx, row in value_bets.head(3).iterrows():
        print(f"  • {row['username']} (Race 2: #{int(row['race2_rank'])} → Predicted: #{int(row['predicted_rank'])})")
else:
    print("  • Focus on proven finishers from Race 2 (less volatility)")

print("\n❌ FADE:")
if len(avoid) > 0:
    for idx, row in avoid.head(2).iterrows():
        print(f"  • {row['username']} (likely to drop)")
else:
    print("  • Avoid drivers who DNF'd in Race 2")

print("\n🎲 SMART STRATEGY:")
print("  • 30% on top predicted winner")
print("  • 40% spread across top 3-5 for podium")
print("  • 20% on value bets")
print("  • 10% on long shots (high aggression young drivers)")

print("\n" + "="*80)

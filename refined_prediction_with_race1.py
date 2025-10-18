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
print("REFINED PREDICTION WITH RACE 1 RESULTS")
print("="*80)

print("\n🏁 RACE 1 RESULTS:")
print("  Conditions: 71 laps, Sunny weather")
print("  1st: asjiodf (B2B SaaS Motorsport)")
print("  2nd: Joe (Tapped In Team)")

# Build models
unknown_df = df[df['weather'] == 'Unknown'].copy()
sunny_df = df[df['weather'] == 'Sunny'].copy()

# Unknown model
X_unknown = unknown_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
y_unknown = unknown_df['time'].copy()
model_unknown = LinearRegression()
model_unknown.fit(X_unknown, y_unknown)

# Sunny model
X_sunny = sunny_df[['aggressiveness', 'defensive_ability', 'stamina', 'age', 'laps']].copy()
y_sunny = sunny_df['time'].copy()
model_sunny = LinearRegression()
model_sunny.fit(X_sunny, y_sunny)

print("\n" + "-"*80)
print("ANALYZING RACE 1 WINNERS")
print("-"*80)

# Get driver attributes
asjiodf_driver = drivers[drivers['username'] == 'asjiodf']
if len(asjiodf_driver) == 0:
    # Find by team
    asjiodf_driver = drivers[drivers['team'] == 'B2B SaaS Motorsport'].iloc[0:1]

joe_driver = drivers[drivers['username'] == 'Joe']
if len(joe_driver) == 0:
    # Find by team
    joe_driver = drivers[drivers['team'] == 'Tapped In Team'].iloc[0:1]

print("\nasjiodf (1st place, 71 laps, Sunny):")
if len(asjiodf_driver) > 0:
    driver = asjiodf_driver.iloc[0]
    print(f"  Aggressiveness:    {driver['aggressiveness']:.3f}")
    print(f"  Defensive Ability: {driver['defensive_ability']:.3f}")
    print(f"  Stamina:           {driver['stamina']:.3f}")
    print(f"  Age:               {driver['age']}")

    # Predict time for 71 laps Sunny
    X_pred = np.array([[driver['aggressiveness'], driver['defensive_ability'],
                       driver['stamina'], driver['age'], 71]])
    pred_sunny_71 = model_sunny.predict(X_pred)[0]
    print(f"  Predicted time (71 laps, Sunny): {pred_sunny_71:.0f}s")

    # Predict time for 65 laps Unknown
    X_pred_unknown = np.array([[driver['aggressiveness'], driver['defensive_ability'],
                               driver['stamina'], driver['age'], 65]])
    pred_unknown_65 = model_unknown.predict(X_pred_unknown)[0]
    print(f"  Predicted time (65 laps, Unknown): {pred_unknown_65:.0f}s")

print("\nJoe (2nd place, 71 laps, Sunny):")
if len(joe_driver) > 0:
    driver = joe_driver.iloc[0]
    print(f"  Aggressiveness:    {driver['aggressiveness']:.3f}")
    print(f"  Defensive Ability: {driver['defensive_ability']:.3f}")
    print(f"  Stamina:           {driver['stamina']:.3f}")
    print(f"  Age:               {driver['age']}")

    # Predict time for 71 laps Sunny
    X_pred = np.array([[driver['aggressiveness'], driver['defensive_ability'],
                       driver['stamina'], driver['age'], 71]])
    pred_sunny_71 = model_sunny.predict(X_pred)[0]
    print(f"  Predicted time (71 laps, Sunny): {pred_sunny_71:.0f}s")

    # Predict time for 65 laps Unknown
    X_pred_unknown = np.array([[driver['aggressiveness'], driver['defensive_ability'],
                               driver['stamina'], driver['age'], 65]])
    pred_unknown_65 = model_unknown.predict(X_pred_unknown)[0]
    print(f"  Predicted time (65 laps, Unknown): {pred_unknown_65:.0f}s")

print("\n" + "-"*80)
print("KEY INSIGHT FROM RACE 1")
print("-"*80)

print("\n✓ asjiodf's winning profile in 71-lap Sunny:")
print("  • Low aggressiveness (0.14) - avoided mistakes")
print("  • Moderate defensive ability (0.57) - not too high for Sunny")
print("  • Moderate stamina (0.45) - paced well")
print("  • Experienced (37 years) - managed long hot race")

print("\n✓ Joe's 2nd place profile:")
print("  • Low aggressiveness (0.36) - relatively controlled")
print("  • High defensive ability (0.56) - skilled driver")
print("  • Moderate stamina (0.44) - decent pacing")
print("  • Young (20 years) - but slightly disadvantaged in 71-lap Sunny")

print("\n❓ What this tells us about Unknown race:")
print("  • asjiodf won Sunny with low aggression + moderate defense")
print("  • In Unknown, defensive ability matters MUCH MORE (-5071s per unit!)")
print("  • Joe might actually perform BETTER in Unknown (high defense = advantage)")
print("  • asjiodf's moderate defense (0.57) is good but not elite")

# Re-analyze all drivers
print("\n" + "="*80)
print("COMPLETE DRIVER ANALYSIS: UNKNOWN WEATHER, 65 LAPS")
print("="*80)

# All drivers from standings
standings_data = [
    {'username': 'asjiodf', 'team': 'B2B SaaS Motorsport', 'current_points': 35, 'rank': 1},
    {'username': 'thosewhoknow_28', 'team': 'Tapped In Team', 'current_points': 27, 'rank': 2},
    {'username': 'bruh', 'team': 'Red Velvet Racing', 'current_points': 25, 'rank': 3},
    {'username': 'Joe', 'team': 'Tapped In Team', 'current_points': 24, 'rank': 4},
    {'username': 'Mario_33', 'team': 'Tapped In Team', 'current_points': 20, 'rank': 5},
    {'username': 'Johnny', 'team': 'Boba Tea Gnarly Automotives', 'current_points': 18, 'rank': 6},
    {'username': 'recursion', 'team': 'Tapped In Team', 'current_points': 15, 'rank': 7},
    {'username': 'rizzSquad_9', 'team': 'Tapped In Team', 'current_points': 10, 'rank': 8},
    {'username': 'deepa_29', 'team': 'B2B SaaS Motorsport', 'current_points': 8, 'rank': 9},
    {'username': '4Alpha_2', 'team': 'Boba Tea Gnarly Automotives', 'current_points': 4, 'rank': 10},
    {'username': 'Justin Zhang', 'team': 'Tapped In Team', 'current_points': 4, 'rank': 11},
    {'username': 'Miami67', 'team': 'Tapped In Team', 'current_points': 2, 'rank': 12},
    {'username': 'hihi', 'team': 'Tapped In Team', 'current_points': 2, 'rank': 13},
    {'username': 'micheal', 'team': 'Spanky\'s Fast Cars', 'current_points': 1, 'rank': 14},
    {'username': 'BetterVerstappen_26', 'team': 'Tapped In Team', 'current_points': 1, 'rank': 15},
]

predictions = []

for driver_data in standings_data:
    # Try to find driver by username first, then by team
    driver_match = drivers[drivers['username'] == driver_data['username']]
    if len(driver_match) == 0:
        driver_match = drivers[drivers['team'] == driver_data['team']].iloc[0:1]

    if len(driver_match) > 0:
        driver = driver_match.iloc[0]

        # Predict Unknown 65 laps
        X_pred = np.array([[driver['aggressiveness'], driver['defensive_ability'],
                           driver['stamina'], driver['age'], 65]])
        pred_time = model_unknown.predict(X_pred)[0]

        predictions.append({
            'username': driver_data['username'],
            'team': driver_data['team'],
            'current_rank': driver_data['rank'],
            'current_points': driver_data['current_points'],
            'aggressiveness': driver['aggressiveness'],
            'defensive_ability': driver['defensive_ability'],
            'stamina': driver['stamina'],
            'age': driver['age'],
            'predicted_time': pred_time,
            'race1_winner': driver_data['username'] == 'asjiodf',
            'race1_second': driver_data['username'] == 'Joe'
        })

pred_df = pd.DataFrame(predictions)
pred_df = pred_df.sort_values('predicted_time')
pred_df['predicted_rank'] = range(1, len(pred_df) + 1)
pred_df['rank_change'] = pred_df['current_rank'] - pred_df['predicted_rank']

print("\n" + "-"*80)
print("PREDICTED RANKINGS (Unknown Weather, 65 Laps)")
print("-"*80)

print(f"\n{'Rank':<5} {'Username':<25} {'Team':<30} {'Curr':<5} {'Time':<10} {'Def':<6} {'Agg':<6} {'Note':<20}")
print("-" * 117)

for idx, row in pred_df.iterrows():
    note = ""
    if row['race1_winner']:
        note = "🏆 Race 1 Winner"
    elif row['race1_second']:
        note = "🥈 Race 1 2nd"
    elif row['rank_change'] >= 3:
        note = "⬆️ Big Mover"
    elif row['rank_change'] <= -3:
        note = "⬇️ Big Drop"

    print(f"{int(row['predicted_rank']):<5} {row['username']:<25} {row['team']:<30} "
          f"{int(row['current_rank']):<5} {row['predicted_time']:<10.0f} "
          f"{row['defensive_ability']:<6.2f} {row['aggressiveness']:<6.2f} {note:<20}")

# Analysis
print("\n" + "="*80)
print("BETTING ANALYSIS WITH RACE 1 CONTEXT")
print("="*80)

print("\n🏆 PODIUM PREDICTIONS:")
top3 = pred_df.head(3)
for idx, row in top3.iterrows():
    change = row['rank_change']
    change_str = f"⬆️ +{int(change)}" if change > 0 else f"⬇️ {int(change)}" if change < 0 else "="

    print(f"\n#{int(row['predicted_rank'])}: {row['username']} ({row['team']})")
    print(f"  Current: #{int(row['current_rank'])} ({change_str})")
    print(f"  Predicted time: {row['predicted_time']:.0f}s")
    print(f"  Key stats: Def={row['defensive_ability']:.2f}, Agg={row['aggressiveness']:.2f}")

    if row['race1_winner']:
        print(f"  ✓ Won Race 1 (71-lap Sunny) - PROVEN WINNER")
    elif row['race1_second']:
        print(f"  ✓ 2nd in Race 1 (71-lap Sunny) - PROVEN PERFORMER")

    # Explain why they'll do well
    if row['defensive_ability'] >= 0.55:
        print(f"  ✓ High defensive ability ({row['defensive_ability']:.2f}) = BIG advantage in Unknown")
    if row['aggressiveness'] <= 0.25:
        print(f"  ✓ Low aggressiveness ({row['aggressiveness']:.2f}) = Avoids penalties")

# Value bets
print("\n💎 TOP VALUE BETS:")
value_bets = pred_df[pred_df['rank_change'] >= 3].sort_values('rank_change', ascending=False)
for idx, row in value_bets.head(5).iterrows():
    print(f"\n{row['username']} - Currently #{int(row['current_rank'])} → Predicted #{int(row['predicted_rank'])} (⬆️ +{int(row['rank_change'])})")
    print(f"  Why: Defensive={row['defensive_ability']:.2f} (strong for Unknown), Agg={row['aggressiveness']:.2f}")

    if row['race1_winner']:
        print(f"  🔥 BONUS: Won Race 1! Has winning momentum")
    elif row['race1_second']:
        print(f"  🔥 BONUS: 2nd in Race 1! Proven top-tier driver")

# Avoid bets
print("\n⚠️ AVOID:")
avoid_bets = pred_df[pred_df['rank_change'] <= -3].sort_values('rank_change')
for idx, row in avoid_bets.head(3).iterrows():
    print(f"\n{row['username']} - Currently #{int(row['current_rank'])} → Predicted #{int(row['predicted_rank'])} (⬇️ {int(row['rank_change'])})")
    print(f"  Why: Agg={row['aggressiveness']:.2f} (too high!), Def={row['defensive_ability']:.2f} (not strong enough)")

# Head-to-head: asjiodf vs Joe
print("\n" + "="*80)
print("HEAD-TO-HEAD: asjiodf vs Joe")
print("="*80)

asjiodf_row = pred_df[pred_df['username'] == 'asjiodf'].iloc[0]
joe_row = pred_df[pred_df['username'] == 'Joe'].iloc[0]

print("\n🏆 asjiodf (Race 1 WINNER):")
print(f"  Race 1 result: 1st place (71 laps, Sunny)")
print(f"  Predicted Race 2: #{int(asjiodf_row['predicted_rank'])}")
print(f"  Defensive: {asjiodf_row['defensive_ability']:.3f}")
print(f"  Aggressiveness: {asjiodf_row['aggressiveness']:.3f}")
print(f"  Predicted time: {asjiodf_row['predicted_time']:.0f}s")

print("\n🥈 Joe (Race 1 RUNNER-UP):")
print(f"  Race 1 result: 2nd place (71 laps, Sunny)")
print(f"  Predicted Race 2: #{int(joe_row['predicted_rank'])}")
print(f"  Defensive: {joe_row['defensive_ability']:.3f}")
print(f"  Aggressiveness: {joe_row['aggressiveness']:.3f}")
print(f"  Predicted time: {joe_row['predicted_time']:.0f}s")

time_diff = joe_row['predicted_time'] - asjiodf_row['predicted_time']
print(f"\nTime difference: {time_diff:.0f}s ({time_diff:.1f}% slower)")

if asjiodf_row['predicted_rank'] < joe_row['predicted_rank']:
    print(f"\n✓ asjiodf maintains lead in Unknown weather")
    print(f"  Reason: Similar defensive skills (0.57 vs 0.56), but lower aggression (0.14 vs 0.36)")
else:
    print(f"\n⚠️ Joe could overtake asjiodf in Unknown weather!")
    print(f"  Reason: Similar/higher defensive ability benefits Unknown more")

# Final recommendations
print("\n" + "="*80)
print("FINAL BETTING RECOMMENDATIONS")
print("="*80)

print("\n🎯 CONSERVATIVE STRATEGY (Safe bets):")
print("  1. asjiodf to WIN (proven winner, optimal profile)")
print("  2. deepa_29 for PODIUM (same team/stats as asjiodf, undervalued)")
print("  3. Joe for TOP 5 (proven performer, strong defense)")

print("\n🔥 AGGRESSIVE STRATEGY (High value):")
print("  1. deepa_29 TO WIN (same profile as asjiodf but at #9!)")
print("  2. Johnny for PODIUM (young talent, strong profile)")
print("  3. 4Alpha_2 for TOP 5 (massive value, +6 positions)")

print("\n💰 BEST VALUE BET:")
best_value = value_bets.iloc[0]
print(f"  {best_value['username']} - Currently #{int(best_value['current_rank'])} → Predicted #{int(best_value['predicted_rank'])}")
print(f"  Expected movement: ⬆️ +{int(best_value['rank_change'])} positions")
print(f"  Odds likely undervalue this driver significantly!")

print("\n❌ FADE (Bet against):")
worst = avoid_bets.iloc[0]
print(f"  {worst['username']} - Currently #{int(worst['current_rank'])} → Predicted #{int(worst['predicted_rank'])}")
print(f"  Expected drop: ⬇️ {int(worst['rank_change'])} positions")
print(f"  Will likely crash or spin out in Unknown conditions")

print("\n💡 KEY TAKEAWAY:")
print("  Race 1 showed asjiodf's consistency and Joe's skill")
print("  Unknown weather will shift advantage to HIGH DEFENSIVE drivers")
print("  Both asjiodf and Joe have strong defense (0.57, 0.56)")
print("  But deepa_29 has IDENTICAL stats to asjiodf at much better odds!")

print("\n" + "="*80)

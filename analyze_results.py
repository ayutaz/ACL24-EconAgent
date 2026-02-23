"""フルシミュレーション結果の分析スクリプト"""
import pickle as pkl
import numpy as np

data_path = 'data/gpt-3-noperception-reflection-1-100agents-240months'

# Load final data
with open(f'{data_path}/dense_log.pkl', 'rb') as f:
    dense_log = pkl.load(f)

with open(f'{data_path}/env_240.pkl', 'rb') as f:
    env = pkl.load(f)

num_agents = env.num_agents
actions_log = dense_log['actions']
states_log = dense_log['states']

print('=' * 70)
print('  EconAgent Full Simulation Results (100 agents / 240 months)')
print('=' * 70)

# === 1. Macro Indicators ===
print('\n=== 1. Macro Economic Indicators ===\n')

prices = env.world.price
wages = env.world.wage
interest_rates = env.world.interest_rate

print('--- Price ---')
print('  Initial: {:.2f}'.format(prices[0]))
print('  Final:   {:.2f}'.format(prices[-1]))
print('  Change:  {:.1f}%'.format((prices[-1] / prices[0] - 1) * 100))
print('  Min:     {:.2f} (Month {})'.format(min(prices), prices.index(min(prices))))
print('  Max:     {:.2f} (Month {})'.format(max(prices), prices.index(max(prices))))

print('\n--- Wage ---')
print('  Initial: {:.2f}'.format(wages[0]))
print('  Final:   {:.2f}'.format(wages[-1]))
print('  Change:  {:.1f}%'.format((wages[-1] / wages[0] - 1) * 100))

print('\n--- Interest Rate ---')
print('  Values:  {}'.format([round(r, 4) for r in interest_rates]))

print('\n--- GDP ---')
print('  Nominal GDP: {:>15,.0f}'.format(env.world.nominal_gdp[0] if env.world.nominal_gdp else 0))
print('  Real GDP:    {:>15,.0f}'.format(env.world.real_gdp[0] if env.world.real_gdp else 0))

# Yearly price inflation
print('\n--- Yearly Price Inflation ---')
for year in range(20):
    start_month = year * 12
    end_month = (year + 1) * 12
    if end_month < len(prices):
        yearly_inf = (prices[end_month] / prices[start_month] - 1) * 100
        print('  Year {:>2} (Month {:>3}-{:>3}): {:>+7.2f}%'.format(year + 1, start_month + 1, end_month, yearly_inf))

# === 2. Monthly Work Rate & Consumption ===
print('\n=== 2. Quarterly Work Rate & Avg Consumption ===\n')
print('{:>8} | {:>9} | {:>12} | {:>8} | {:>8}'.format('Quarter', 'WorkRate', 'AvgConsumption', 'Price', 'Wage'))
print('-' * 55)

for q in range(80):  # 240 months / 3 = 80 quarters
    months = range(q * 3, min((q + 1) * 3, len(actions_log)))
    work_counts = []
    consume_vals = []
    for m in months:
        step_actions = actions_log[m]
        wc = 0
        for idx in range(num_agents):
            a = step_actions.get(str(idx), {})
            if a.get('SimpleLabor', 0) > 0:
                wc += 1
            consume_vals.append(a.get('SimpleConsumption', 0))
        work_counts.append(wc)
    avg_wr = np.mean(work_counts) / num_agents * 100
    avg_c = np.mean(consume_vals) if consume_vals else 0
    mid_month = q * 3 + 1
    price = prices[mid_month] if mid_month < len(prices) else 0
    wage = wages[mid_month] if mid_month < len(wages) else 0
    if q % 4 == 0:  # Print every 4th quarter (yearly)
        print('  Q{:>3}   | {:>8.1f}% | {:>12.1f} | {:>8.2f} | {:>8.2f}'.format(q + 1, avg_wr, avg_c, price, wage))

# === 3. Unemployment ===
print('\n=== 3. Unemployment (sampled yearly) ===\n')
for year in range(20):
    month = (year + 1) * 12
    if month < len(states_log):
        unemploy = sum(1 for idx in range(num_agents)
                       if states_log[month][str(idx)]['endogenous']['job'] == 'Unemployment')
        print('  Year {:>2} (Month {:>3}): {:>3} unemployed ({:.0f}%)'.format(
            year + 1, month, unemploy, unemploy / num_agents * 100))

# === 4. Wealth Distribution ===
print('\n=== 4. Wealth Distribution (Final) ===\n')
wealths = [env.get_agent(str(i)).inventory['Coin'] for i in range(num_agents)]
wealths_sorted = sorted(wealths)

print('  Total:   {:>15,.0f}'.format(sum(wealths)))
print('  Mean:    {:>15,.0f}'.format(np.mean(wealths)))
print('  Median:  {:>15,.0f}'.format(np.median(wealths)))
print('  Min:     {:>15,.0f}'.format(min(wealths)))
print('  Max:     {:>15,.0f}'.format(max(wealths)))
print('  Std Dev: {:>15,.0f}'.format(np.std(wealths)))

# Gini coefficient
n = len(wealths)
gini = np.sum(np.abs(np.subtract.outer(wealths, wealths))) / (2 * n * sum(wealths))
print('  Gini:    {:>15.4f}'.format(gini))

# Quintile breakdown
print('\n  --- Wealth by Quintile ---')
for q in range(5):
    start = q * 20
    end = (q + 1) * 20
    quintile = wealths_sorted[start:end]
    labels = ['Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%']
    print('  {:>10}: Mean {:>12,.0f} | Share {:>5.1f}%'.format(
        labels[q], np.mean(quintile), sum(quintile) / sum(wealths) * 100))

# === 5. Top 10 & Bottom 10 Agents ===
print('\n=== 5. Top 10 Wealthiest Agents ===\n')
agent_data = []
for idx in range(num_agents):
    agent = env.get_agent(str(idx))
    work_months = sum(1 for a in actions_log if a.get(str(idx), {}).get('SimpleLabor', 0) > 0)
    agent_data.append({
        'idx': idx,
        'name': agent.endogenous['name'],
        'age': agent.endogenous['age'],
        'job': agent.endogenous['job'],
        'skill': agent.state['skill'],
        'wealth': agent.inventory['Coin'],
        'work_months': work_months,
    })

agent_data.sort(key=lambda x: x['wealth'], reverse=True)

print('{:>3} | {:<22} | {:>3} | {:<25} | {:>6} | {:>12} | {:>5}'.format(
    '#', 'Name', 'Age', 'Job', 'Skill', 'Wealth', 'Work'))
print('-' * 90)
for i, a in enumerate(agent_data[:10]):
    print('{:>3} | {:<22} | {:>3} | {:<25} | {:>6.1f} | {:>12,.0f} | {:>3}/240'.format(
        i + 1, a['name'], a['age'], a['job'], a['skill'], a['wealth'], a['work_months']))

print('\n=== 6. Bottom 10 Agents ===\n')
print('{:>3} | {:<22} | {:>3} | {:<25} | {:>6} | {:>12} | {:>5}'.format(
    '#', 'Name', 'Age', 'Job', 'Skill', 'Wealth', 'Work'))
print('-' * 90)
for i, a in enumerate(agent_data[-10:]):
    print('{:>3} | {:<22} | {:>3} | {:<25} | {:>6.1f} | {:>12,.0f} | {:>3}/240'.format(
        91 + i, a['name'], a['age'], a['job'], a['skill'], a['wealth'], a['work_months']))

# === 6. Skill vs Wealth correlation ===
print('\n=== 7. Skill-Wealth Correlation ===\n')
skills = [a['skill'] for a in agent_data]
wealth_vals = [a['wealth'] for a in agent_data]
corr = np.corrcoef(skills, wealth_vals)[0, 1]
print('  Pearson correlation: {:.4f}'.format(corr))

# === 7. Work participation over time ===
print('\n=== 8. Work Participation Summary ===\n')
work_months_all = [a['work_months'] for a in agent_data]
print('  Mean work months:   {:.1f}/240'.format(np.mean(work_months_all)))
print('  Median work months: {:.0f}/240'.format(np.median(work_months_all)))
print('  Min work months:    {}/240'.format(min(work_months_all)))
print('  Max work months:    {}/240'.format(max(work_months_all)))

print('\n' + '=' * 70)
print('  Analysis Complete')
print('=' * 70)

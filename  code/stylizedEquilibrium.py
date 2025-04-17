import numpy as np
import matplotlib.pyplot as plt

# Model Parameters
alpha = 0.5       # Cobb-Douglas utility share
I = 100           # Consumer income
c = 1             # Marginal cost coefficient (quadratic)
p_world = 6       # World price (lower than autarky)
tau = 0.5         # Tariff rate (50%)

# Derived prices
p_tariff = p_world * (1 + tau)

# Marshallian demand: q = αI / p
def demand(p):
    return alpha * I / p

# Profit-maximizing supply: MC = P ⇒ dC/dq = 2cq = p ⇒ q = p / (2c)
def supply(p):
    return p / (2 * c)

# Equilibrium quantities
q_d_free = demand(p_world)
q_s_free = supply(p_world)
imports_free = q_d_free - q_s_free

q_d_tariff = demand(p_tariff)
q_s_tariff = supply(p_tariff)
imports_tariff = q_d_tariff - q_s_tariff

# Welfare calculations
def consumer_surplus(p, q):
    return alpha * I * np.log(q) - p * q

def producer_surplus(p, q):
    return p * q - c * q**2

cs_free = consumer_surplus(p_world, q_d_free)
ps_free = producer_surplus(p_world, q_s_free)

cs_tariff = consumer_surplus(p_tariff, q_d_tariff)
ps_tariff = producer_surplus(p_tariff, q_s_tariff)
gov_rev = p_world * tau * imports_tariff

ts_free = cs_free + ps_free
ts_tariff = cs_tariff + ps_tariff + gov_rev
dwl = ts_free - ts_tariff

# Plotting
q_vals = np.linspace(0.01, q_d_free * 1.2, 500)
p_demand_vals = alpha * I / q_vals
p_supply_vals = 2 * c * q_vals

plt.figure(figsize=(10, 6))
plt.plot(q_vals, p_demand_vals, label="Demand", color="blue")
plt.plot(q_vals, p_supply_vals, label="Supply", color="green")
plt.axhline(p_world, color='gray', linestyle='--', label=f"World Price (${p_world:.2f})")
plt.axhline(p_tariff, color='black', linestyle='--', label=f"Tariff Price (${p_tariff:.2f})")

# Welfare shading under tariff
plt.fill_between(q_vals[q_vals <= q_d_tariff],
                 alpha * I / q_vals[q_vals <= q_d_tariff],
                 p_tariff, color='blue', alpha=0.2, label='Consumer Surplus')
plt.fill_between(q_vals[q_vals <= q_s_tariff],
                 p_tariff, 2 * c * q_vals[q_vals <= q_s_tariff],
                 color='green', alpha=0.2, label='Producer Surplus')
plt.fill_between([q_s_tariff, q_d_tariff],
                p_world, p_tariff,
                color='red', alpha=0.2, label='Tariff Revenue')
# plt.fill_between([q_s_tariff, q_d_tariff],
#                  p_tariff, p_demand_vals[(q_vals > q_s_tariff) & (q_vals < q_d_tariff)],
#                  color='gray', alpha=0.2, label='Deadweight Loss')

# Annotations
#plt.axvline(q_s_tariff, color='green', linestyle=':', label="Domestic Supply (Tariff)")
#plt.axvline(q_d_tariff, color='blue', linestyle=':', label="Domestic Demand (Tariff)")
#plt.axvline(q_d_free, color='blue', linestyle='--', alpha=0.4)
#plt.axvline(q_s_free, color='green', linestyle='--', alpha=0.4)

plt.title("Figure 1: Partial Equilibrium with Tariff")
plt.xlabel("Quantity")
plt.ylabel("Price")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.xlim(1,10)
plt.ylim(0,30)
plt.savefig('../output/figures/figure1.png')

# Print welfare results
print(f"--- Welfare Analysis ---")
print(f"Consumer Surplus (Free Trade):     {cs_free:.2f}")
print(f"Producer Surplus (Free Trade):     {ps_free:.2f}")
print(f"Total Surplus (Free Trade):        {ts_free:.2f}")
print()
print(f"Consumer Surplus (With Tariff):    {cs_tariff:.2f}")
print(f"Producer Surplus (With Tariff):    {ps_tariff:.2f}")
print(f"Government Revenue:                {gov_rev:.2f}")
print(f"Total Surplus (With Tariff):       {ts_tariff:.2f}")
print(f"Deadweight Loss:                   {dwl:.2f}")

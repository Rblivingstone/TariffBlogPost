import numpy as np
import matplotlib.pyplot as plt
from estimation import estimate
from scipy.optimize import root_scalar

def run(tau, params):
    p_world = 1       # World price (lower than autarky)
    alpha = params[0]
    I = 3
    a = params[1]
    b = params[2]
    # Derived prices
    p_tariff = p_world * (1 + tau)

    # Marshallian demand: q = αI / p
    def demand(p):
        return alpha * I / p

    # Profit-maximizing supply: MC = P ⇒ dC/dq = 2cq = p ⇒ q = p / (2c)
    def supply(p):
        return (p-b)/a


    def excess_demand(p):
        return demand(p) - supply(p)

    star = root_scalar(excess_demand, bracket=[1, 100], method='bisect')
    print(f"P_star: {star.root}")
    pstar = star.root
    Qstar = demand(pstar)
    print(f"Q_star: {Qstar}")


    # Equilibrium quantities
    q_d_free = demand(p_world)
    q_s_free = supply(p_world)
    imports_free = q_d_free - q_s_free

    #producer_burden = (q_s_free-q_star)*p_world -

    q_d_tariff = demand(p_tariff)
    q_s_tariff = supply(p_tariff)
    imports_tariff = q_d_tariff - q_s_tariff

    consumer_burden = alpha * I * np.log(q_d_free) - alpha * I * np.log(q_d_tariff) - (q_d_free - q_d_tariff) * p_world + (p_tariff-p_world)*(q_d_tariff-Qstar)

    producer_burden = 0.5*(p_tariff-p_world)*(q_s_tariff-q_s_free)+(p_tariff-p_world)*(Qstar-q_s_tariff)

    cburden = 100*consumer_burden/(consumer_burden+producer_burden)
    pburden = 100*producer_burden / (consumer_burden + producer_burden)
    # Welfare calculations
    def consumer_surplus(p, q):
        return alpha * I * np.log(q)-alpha*I*np.log(0.01) - p * q

    def producer_surplus(p, q):
        return p * q - a/2 * q**2 -b*q

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
    p_supply_vals = a * q_vals+b

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
                     p_tariff, a * q_vals[q_vals <= q_s_tariff]+b,
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

    plt.title("Figure 3: Estimated Partial Equilibrium with Tariff")
    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0,2)
    plt.ylim(0,4)
    plt.savefig('../output/figures/figure3.png')

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
    print()
    print(f"Consumer Burden Share:                   {cburden:.2f}%")
    print(f"Producer Burden Share:                   {pburden:.2f}%")

if __name__=='__main__':
    run(.0147, estimate())
    run(0.29, estimate())
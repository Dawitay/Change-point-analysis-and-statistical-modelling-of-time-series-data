import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

# Load and prepare data (same as before)
df = pd.read_csv("../srv.BrentOilPrices.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['LogPrice'] = np.log(df['Price'])
df['LogReturn'] = df['LogPrice'].diff()
df = df.dropna(subset=['LogReturn'])

returns = df['LogReturn'].values
dates = df['Date'].values
n = len(returns)

# Load or re-train the model
with pm.Model() as model:
    tau = pm.DiscreteUniform('tau', lower=0, upper=n)
    mu_1 = pm.Normal('mu_1', mu=np.mean(returns), sigma=1)
    mu_2 = pm.Normal('mu_2', mu=np.mean(returns), sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)
    idx = np.arange(n)
    mu = pm.math.switch(tau >= idx, mu_1, mu_2)
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=returns)
    
    trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.95)

# Check for convergence using r_hat and trace
print("\n📊 Summary of Posterior Distributions:")
summary = az.summary(trace, var_names=['tau', 'mu_1', 'mu_2', 'sigma'])
print(summary)

# Check that r_hat ≈ 1.0 for all parameters
print("\n✅ Check r_hat values: Should be close to 1.0 for convergence.\n")

# Trace plots
az.plot_trace(trace, var_names=['tau', 'mu_1', 'mu_2', 'sigma'])
plt.tight_layout()
plt.show()

#  Plot posterior of change point tau

plt.figure(figsize=(10, 4))
az.plot_posterior(trace, var_names=["tau"], hdi_prob=0.95)
plt.title('Posterior Distribution of Change Point (τ)')
plt.xlabel('Index in Time Series')
plt.grid(True)
plt.show()

# Get estimated change date
tau_est = int(trace.posterior['tau'].mean().values)
change_date = dates[tau_est]
print(f"\n📍 Estimated change point: Index {tau_est} → Date: {change_date.date()}")

#  Compare before and after means (mu_1 vs mu_2)
az.plot_posterior(trace, var_names=['mu_1', 'mu_2'], hdi_prob=0.95)
plt.suptitle('Posterior Distributions of Mean Returns Before and After Change Point')
plt.show()

# Probabilistic comparison (mu_2 > mu_1)
mu_1_samples = trace.posterior['mu_1'].values.flatten()
mu_2_samples = trace.posterior['mu_2'].values.flatten()
prob_mu2_greater = np.mean(mu_2_samples > mu_1_samples)

print(f"\n📈 Probability that mean after change > mean before change: {prob_mu2_greater:.2%}")

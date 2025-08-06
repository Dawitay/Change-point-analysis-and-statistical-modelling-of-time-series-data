import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az

# Load and preprocess the data

df = pd.read_csv("../srv/BrentOilPrices.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Compute log prices and log returns
df['LogPrice'] = np.log(df['Price'])
df['LogReturn'] = df['LogPrice'].diff()
df = df.dropna(subset=['LogReturn'])  # Remove NaN in first row

# Extract required data for modeling
returns = df['LogReturn'].values
dates = df['Date'].values
n = len(returns)  # Number of observations

# Bayesian Change Point Model
with pm.Model() as model:
    # a. Define the switch point (tau): unknown index of structural change
    tau = pm.DiscreteUniform('tau', lower=0, upper=n)

    # b. Define the "before" and "after" means
    mu_1 = pm.Normal('mu_1', mu=np.mean(returns), sigma=1)
    mu_2 = pm.Normal('mu_2', mu=np.mean(returns), sigma=1)

    # Shared standard deviation
    sigma = pm.HalfNormal('sigma', sigma=1)

    # c. Switch function: choose mu_1 if index < tau, else mu_2
    idx = np.arange(n)
    mu = pm.math.switch(tau >= idx, mu_1, mu_2)

    # d. Likelihood: observed data follows Normal distribution with the chosen mu
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=returns)

    # e. Run MCMC sampler to draw from posterior
    trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True, target_accept=0.95)



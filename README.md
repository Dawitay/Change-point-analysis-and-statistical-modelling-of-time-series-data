## 📈 Brent Oil Price Change Point Detection – Project Summary

This notebook explores **structural shifts in Brent oil prices** by applying both **frequentist** and **Bayesian change point detection** techniques. The goal is to identify periods of significant change in either the **price level** or **volatility**, and to interpret those shifts in the context of real-world events.

---

### 🧪 Key Components

#### ✅ 1. **Data Loading and Preprocessing**

* Brent oil price data is cleaned and missing values are interpolated.
* Log returns are computed to capture changes in volatility.

#### ✅ 2. **Frequentist Change Point Detection**

* Uses the `ruptures` library with the PELT algorithm to detect abrupt shifts in **mean price**.
* Change points are plotted over time to visualize potential structural breaks.

#### ✅ 3. **Bayesian Change Point Detection (PyMC)**

* Applies a probabilistic model using PyMC to detect shifts in **volatility**.
* Models the log returns and estimates the most probable change point (`τ`) using MCMC sampling.
* Posterior plots help identify the most likely time of volatility regime change.

#### ✅ 4. **Interpretation and Conceptual Analysis**

* Results from both models are interpreted with a focus on **economic or geopolitical events** that may explain the shifts.
* The notebook prepares the ground for associating these shifts with real-world events from an external `Events.csv` file (to be merged in future work).

---

### 📌 Technologies Used

* Python, Pandas, NumPy
* `ruptures` for frequentist change detection
* `PyMC` + `ArviZ` for Bayesian inference and posterior visualization
* Matplotlib for plotting

---

# Reliable Option Pricing Anomaly Score

## 1. Overview

 This repository accompanies the manuscript **Reliable option pricing through deep learning: An anomaly score-based approach.**
We detect irregular observations in option data using Isolation Forest and incorporate the resulting anomaly scores as weights in the neural network loss. Rather than discarding data, we down-weight low-reliability samples, which improves robustness especially for short-maturity and low-liquidity contracts.

## **Key ideas**

 __Treat anomaly score as an inverse reliability signal and inject it into training via a weighted MSE__

 __Preserve data coverage (no wholesale deletion) while mitigating the influence of stale/noisy quotes__

 __Aligns with market intuition: anomalies concentrate at very short/long maturities, thin liquidity, and extreme moneyness__


## 2. What’s in this repo
### 2.1 ano_nn_analysis.py 
  - Core training/evaluation script (baseline MLP vs. anomaly-weighted MLP).

### 2.2 anoNN_different_option_characteristics.ipynb 
  - Reproduces results by moneyness, time-to-maturity, and volume buckets.

### 2.3 anoNN_rolling_timeseries_splits.ipynb 
  - Rolling train/test splits to assess robustness over time.

### 2.4 data_visualize.ipynb 
  - Basic EDA and plotting utilities.

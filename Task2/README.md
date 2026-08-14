# Task 2 — Explanation Stability & Consistency

Evaluate how **stable** and **consistent** SHAP and LIME explanations are, using the
same Taiwan credit card default dataset and logistic regression model as Task 1.

## Dataset & Model

[Default of Credit Card Clients — Taiwan (UCI ML Repository, ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
Balanced logistic regression (accuracy 0.68), same 80/20 split and scaling as Task 1
(`random_state=42`).

## What this notebook does

1. Reuse the same dataset and model.
2. Generate SHAP and LIME explanations for multiple test customers.
3. Compare their top-5 features (cross-method consistency).
4. Repeat LIME with 10 random seeds to measure stability, and confirm SHAP is deterministic.



## Results



### 1. Top-5 feature comparison (SHAP vs LIME)


| Customer | Shared | Agreement |
| -------- | ------ | --------- |
| 0        | 4      | 0.8       |
| 1        | 3      | 0.6       |
| 2        | 4      | 0.8       |
| 3        | 1      | 0.2       |
| 4        | 3      | 0.6       |




### 2. LIME stability across 10 random seeds (customer #0)


| Feature     | Appearance rate | Stable? |
| ----------- | --------------- | ------- |
| `pay_0`     | 1.0             | always  |
| `pay_amt1`  | 1.0             | always  |
| `bill_amt1` | 1.0             | always  |
| `limit_bal` | 1.0             | always  |
| `pay_2`     | 0.7             | often   |
| `pay_3`     | 0.2             | rarely  |
| `marriage`  | 0.1             | rarely  |


**SHAP repeated on the same customer -> identical every time (**`np.allclose == True`**).**

## Where SHAP and LIME agree / disagree

- **Agree** on high-importance features: `pay_0`, `limit_bal`, and the recent
bill/payment amounts. Customers 0 and 2 reach 0.8 agreement, with nearly identical
top-5 lists.
- **Disagree** on low-importance features. Customer 3 drops to 0.2: SHAP ranks all
recent bill amounts (`bill_amt1..6`), while LIME spreads across payment and limit
features. These weaker features are where the methods diverge.
- **Stability:** SHAP is fully deterministic. LIME's four strongest features are stable
across every seed, but the 5th slot varies (`pay_2`/`pay_3`/`marriage`), because LIME
fits its local model on randomly sampled points.

**Takeaway:** an explanation is trustworthy when a feature is high-importance **and**
confirmed by both methods and across LIME seeds. Low-importance features from a single
LIME run should be treated with caution.

## Setup & Run

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook stability_analysis.ipynb
```


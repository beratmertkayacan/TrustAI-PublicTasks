# Credit Card Default: SHAP vs LIME

Train a default prediction model and explain the same predictions with **SHAP** and **LIME**, then compare the results.

## Dataset

**Taiwan Credit Card Default** — UCI / OpenML [`data_id=42477`](https://www.openml.org/d/42477)

30,000 credit card clients with repayment history, bill statements, and demographics (April–September 2005).

| Target | Meaning |
|--------|---------|
| `0` | No default next month |
| `1` | Default next month |

OpenML provides 23 feature columns named `x1`–`x23`. These correspond directly to the UCI variables `X1`–`X23` (credit limit, demographics, repayment status, bill amounts, payment amounts). There is no separate ID column in this OpenML version.

## Model

- **Algorithm:** Logistic Regression (`class_weight="balanced"`)
- **Preprocessing:** `StandardScaler` on all features
- Linear model chosen so SHAP `LinearExplainer` gives exact attributions.

## Methods

| Method | What it shows | Output |
|--------|---------------|--------|
| SHAP (`LinearExplainer`) | Global + local feature contributions | Summary plot + waterfall for one client |
| LIME (`LimeTabularExplainer`) | Local explanation for one prediction | Bar chart for top features |
| Comparison | Same client, both methods | Table + side-by-side bar chart |

## Notebook flow

1. Load data, apply correct column mapping, train model
2. SHAP summary and single-client waterfall
3. LIME explanation for the same client
4. Compare SHAP and LIME contributions (sign agreement)
5. Method comparison table

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook SHAPvsLIME.ipynb
```

## Key findings

- Recent repayment status (`pay_0`) and bill/payment amounts are the strongest predictors.
- SHAP and LIME agree on sign for the top features on the example client.
- SHAP is deterministic; LIME can vary slightly between runs due to random sampling.

## SHAP vs LIME (summary)

| | SHAP | LIME |
|---|------|------|
| Basis | Game theory (Shapley) | Local linear approximation |
| Scope | Global + local | Local only |
| Guarantee | Yes (for compatible explainers) | No |
| Stability | Deterministic | Can vary |
| Speed | Slower in general; fast for linear models | Fast |

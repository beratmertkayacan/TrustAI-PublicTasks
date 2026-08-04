# Credit Card Default: SHAP vs LIME

Train a default prediction model and explain the same predictions with **SHAP** and **LIME**, then compare the results.


## Dataset

[Default of Credit Card Clients — UCI ML Repository (id=350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

30,000 credit card clients in Taiwan with payment history, bill statements, and demographics. Target `default payment next month`: `1` = default, `0` = paid.

OpenML provides 23 feature columns named `x1`–`x23`. These correspond directly to the UCI variables `X1`–`X23` (credit limit, demographics, repayment status, bill amounts, payment amounts). 

## Model

- **Algorithm:** Logistic Regression (`class_weight="balanced"`)
- **Preprocessing:** `StandardScaler` on all features
- Linear model chosen so SHAP `LinearExplainer` gives exact attributions.



## Methods


| Method                        | What it shows                        | Output                                  |
| ----------------------------- | ------------------------------------ | --------------------------------------- |
| Model evaluation              | Ranking quality on imbalanced data   | ROC curve + AUC                         |
| SHAP (`LinearExplainer`)      | Global + local feature contributions | Summary plot + waterfall for one client |
| LIME (`LimeTabularExplainer`) | Local explanation for one prediction | Bar chart for top features              |
| Comparison                    | Same client, both methods            | Table + side-by-side bar chart          |




## Key findings

- Recent repayment status (`pay_0`) and bill/payment amounts are the strongest predictors.
- SHAP and LIME agree on sign for the top features on the example client.
- SHAP is deterministic; LIME can vary slightly between runs due to random sampling.



## SHAP vs LIME (summary)


|           | SHAP                                      | LIME                       |
| --------- | ----------------------------------------- | -------------------------- |
| Basis     | Game theory (Shapley)                     | Local linear approximation |
| Scope     | Global + local                            | Local only                 |
| Guarantee | Yes (for compatible explainers)           | No                         |
| Stability | Deterministic                             | Can vary                   |
| Speed     | Slower in general; fast for linear models | Fast                       |



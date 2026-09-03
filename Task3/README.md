# Task 3: Explanation Reliability Benchmark Across Models

Train three models on the same dataset, generate SHAP and LIME explanations for multiple test clients, and benchmark explanation reliability along three axes: top-feature agreement, stability under repeated runs, and robustness under small input perturbations.

## Dataset and model

- **Dataset:** Taiwan Credit Card Default — [UCI ML Repository (id=350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), loaded via OpenML.
- **Models:**
  - Logistic Regression (`class_weight="balanced"`, `random_state=42`) — SHAP via `LinearExplainer`
  - Random Forest (`n_estimators=200`, `random_state=42`) — SHAP via `TreeExplainer`
  - Gradient Boosting (`random_state=42`, balanced via `sample_weight`) — SHAP via `TreeExplainer`
- **Split:** 80/20 train/test, `random_state=42`, stratified — same split and sampled test clients as Task 2.
- **Sample size:** 30 test clients, top-10 features per explanation.



## Model performance


| Model               | Accuracy | AUC   |
| ------------------- | -------- | ----- |
| Logistic Regression | 0.680    | 0.708 |
| Random Forest       | 0.801    | 0.761 |
| Gradient Boosting   | 0.765    | 0.779 |




## Reliability dimensions


| Dimension             | Question                                                                   | Method                                                                                          |
| --------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Top-feature agreement | Do SHAP and LIME point to the same important features for the same client? | overlap@K + sign agreement                                                                      |
| Stability             | Does the explanation change when re-run on the same input?                 | LIME across 6 random seeds vs. reference seed, Jaccard similarity                               |
| Robustness            | Does the explanation change when the input changes slightly?               | Small Gaussian perturbation (scale 0.1) on the scaled features, Jaccard similarity vs. original |




## Results


| Model                   | overlap@K | sign_agreement | LIME stability | SHAP robustness | LIME robustness | **Reliability score** |
| ----------------------- | --------- | -------------- | -------------- | --------------- | --------------- | --------------------- |
| **Logistic Regression** | 0.793     | 1.000          | 0.790          | 0.863           | 0.626           | **0.814**             |
| Gradient Boosting       | 0.660     | 0.945          | 0.703          | 0.681           | 0.535           | 0.705                 |
| Random Forest           | 0.637     | 0.942          | 0.674          | 0.610           | 0.568           | 0.686                 |


Reliability score = unweighted mean of the five metrics above (all already on a 0-1, higher-is-better scale).

## Conclusion

Logistic Regression has the most reliable explanations across every dimension measured, despite having the lowest raw accuracy/AUC. Random Forest has the highest accuracy but the least reliable explanations. Predictive performance and explanation reliability move in opposite directions for this dataset. Across all three models, SHAP is consistently more robust to small input perturbations than LIME, and LIME's stability/agreement degrades as model complexity increases

## Deliverables

Three trained models

SHAP/LIME explanations for multiple samples, per model

Model performance table

Top-feature agreement table

LIME stability results

Explanation robustness results

Final reliability score table

Short conclusion on which model has the most reliable explanations

## Setup

```bash
cd ..   
source .venv/bin/activate
pip install -r Task3/requirements.txt
jupyter notebook Task3/explanation_reliability.ipynb
```


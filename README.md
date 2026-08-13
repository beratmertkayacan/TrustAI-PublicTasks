# TrustAI Public Tasks

Explainable AI tasks on credit default prediction (SHAP, LIME).

**Repository:** https://github.com/beratmertkayacan/TrustAI-PublicTasks

## Tasks

| Task | Folder | Description |  
|------|--------|-------------|
| Task 1 | [`Task1/`](Task1/) | SHAP vs LIME comparison on a single prediction | 
| Task 2 | [`Task2/`](Task2/) | Explanation stability and consistency across multiple samples |  

Each task has its own `README.md` with full details (dataset, model, methods, deliverables).

## Setup

Each task keeps its own `requirements.txt`, scoped to what that task needs:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Task1/requirements.txt
pip install -r Task2/requirements.txt
```

Open the notebook for the task you are working on:

```bash
jupyter notebook Task1/SHAPvsLIME.ipynb
jupyter notebook Task2/explanation_stability.ipynb
```

## Project layout

```
.
├── .venv/             
├── Task1/
│   ├── README.md
│   ├── SHAPvsLIME.ipynb
│   ├── requirements.txt
│   ├── data/            
├── Task2/
│   ├── README.md
│   ├── explanation_stability.ipynb
│   ├── requirements.txt
│   ├── data/
│   └── outputs/
```

All tasks currently use the **Taiwan Credit Card Default** dataset ([UCI ML Repository, id=350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)) and the same logistic regression pipeline, unless a task's own README states otherwise.



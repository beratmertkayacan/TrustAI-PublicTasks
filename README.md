# TrustAI Public Tasks

Explainable AI tasks on credit default prediction (SHAP, LIME).

**Repository:** https://github.com/beratmertkayacan/TrustAI-PublicTasks

## Tasks

| Task | Folder | Description |
|------|--------|-------------|
| Task 1 | [`Task1/`](Task1/) | SHAP vs LIME comparison on a single prediction |
| Task 2 | [`Task2/`](Task2/) | Explanation stability and consistency across multiple samples |
| Task 3 | [`Task3/`](Task3/) | Explanation reliability benchmark across three models (LR, RF, GB) |

Each task has its own `README.md` with full details (dataset, model, methods, deliverables).

## Setup

Each task keeps its own `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r Task1/requirements.txt
pip install -r Task2/requirements.txt
pip install -r Task3/requirements.txt
```

Open the notebook for the task you are working on:

```bash
jupyter notebook Task1/SHAPvsLIME.ipynb
jupyter notebook Task2/explanation_stability.ipynb
jupyter notebook Task3/explanation_reliability.ipynb
```

## Project layout

```
.
├── .venv/
├── Task1/
│   ├── README.md
│   ├── SHAPvsLIME.ipynb
│   └── requirements.txt
├── Task2/
│   ├── README.md
│   ├── explanation_stability.ipynb
│   ├── requirements.txt
│   └── outputs/
└── Task3/
    ├── README.md
    ├── explanation_reliability.ipynb
    ├── requirements.txt
    └── outputs/
```

All tasks use the **Taiwan Credit Card Default** dataset ([UCI ML Repository, id=350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)), loaded via OpenML, with the same 80/20 stratified split (`random_state=42`).

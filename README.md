# HoopsMM — NeurIPS 2025 Codebase  
Multimodal Benchmarks + Baselines for NCAA Division I Men’s Basketball  
> **Core Tasks**:  
> 1. Pre‑game win‑probability forecasting (`forecast_win/`)  
> 2. Risk‑aware capital allocation (`allocate_capital/`)  
> 3. Season‑long team ranking (`rank_teams/`)

---

## 1 · Quick start

```bash
# clone & enter
git clone https://github.com/<user>/HoopsMM_NeurIPS.git
cd HoopsMM_NeurIPS

# install deps
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```

### Training & evaluation (default configs in `configs/`)

| Task | Train | Evaluate |
|------|-------|----------|
| Win‑forecast | `python tasks/forecast_win/train.py --config configs/forecast_win.yaml` | `python tasks/forecast_win/evaluate.py --config configs/forecast_win.yaml` |
| Capital‑allocation | — (uses preds + odds) | `python tasks/allocate_capital/evaluate.py --config configs/allocate_capital.yaml` |
| Team‑ranking | `python tasks/rank_teams/train.py --config configs/rank_teams.yaml` | `python tasks/rank_teams/evaluate.py --config configs/rank_teams.yaml` |

`output/` holds all trained models, logs, and Monte‑Carlo plots.

---

## 2 · Repository layout

```
tasks/
├── forecast_win/          # XGBoost baseline
├── allocate_capital/      # bet record builder + MC simulator
└── rank_teams/            # Gradient‑Boosting baseline
data/
├──
├──
├──
├──
└── 
requirements.txt
```

Each task exposes:

* `data.py`  — ingest & preprocess  
* `model.py` — registry / persistence  
* `train.py` — CLI entry‑point  
* `evaluate.py` — metrics & plots

---

## 3 · Data expectations

| Path | Needed columns |
|------|----------------|
| `Data/merged.csv` | model features, `home_win`, betting‑odds columns |
| `data/ranking_top.csv` | season‑level engineered features, `rank_group`, `season` |

Adjust the `data.path` field in the YAML if your files live elsewhere.

---

## 4 · Config anatomy (YAML)

```yaml
data:
  path: Data/merged.csv      # CSV location
  features: [...]            # for forecast_win
split:                       # temporal split params
  test_frac: 0.2
model:
  params: {}                 # hyper‑parameter overrides
stake:                       # allocate_capital only
  strategy: kelly | flat
simulation:
  num_runs: 100000
training:
  output_dir: output/<task>
```

---

## 5 · Reproducibility

* Deterministic seeds set for all sklearn models.  
* All hyper‑parameters logged through the YAML configs.  
* Monte‑Carlo simulator stores a histogram (`mc_hist.png`) plus a `.txt` summary for audit trails.

---

## 6 · License
Code released under **MIT License**; dataset under **CC BY 4.0** as noted in the repo.

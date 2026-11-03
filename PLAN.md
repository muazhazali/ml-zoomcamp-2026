# MRT Kajang Ridership Forecasting — Project Completion Checklist

A detailed, ordered checklist. Work top to bottom. Each phase ends with a **"Definition of done"** and the rubric criteria it satisfies. Tick boxes as you go.

---

## Phase 0 — Project Setup

- [ ] Create a new GitHub repository, e.g. `mrt-kajang-forecasting`.
- [ ] Initialise locally and push an empty commit so the remote exists.
  ```bash
  git init
  git remote add origin git@github.com:<you>/mrt-kajang-forecasting.git
  git commit --allow-empty -m "init"
  git push -u origin main
  ```
- [ ] Create the folder skeleton:
  ```
  data/  models/  notebooks/  src/  app/  k8s/
  ```
- [ ] Add `.gitignore` covering:
  ```
  .venv/
  __pycache__/
  data/*.parquet
  models/*.joblib
  .ipynb_checkpoints/
  .env
  *.pyc
  ```
- [ ] Add `data/.gitkeep` and `models/.gitkeep` so folders are tracked.
- [ ] Create `requirements.txt` with **pinned** versions once you know them (you can fill this after Phase 3, but stub it now with the known packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `fastapi`, `uvicorn`, `joblib`, `pyarrow`, `jupyter`).
- [ ] Create the virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] Confirm `python --version` and `pip list` look correct.

**Definition of done:** repo exists on GitHub, folder skeleton in place, venv activates cleanly.
**Rubric:** 0/2 → 1/2 on *Dependency and environment management* (needs README instructions to reach 2).

---

## Phase 1 — Data Acquisition

- [ ] Open `https://data.gov.my/data-catalogue/ridership_headline`.
- [ ] Download **Full Dataset → Parquet**.
- [ ] Save it to `data/ridership_headline.parquet`.
- [ ] Write `download_data.py` that either:
  - fetches the Parquet from the URL (use `requests` + write bytes), **or**
  - prints the manual-download instructions clearly.
- [ ] Test `download_data.py` on a clean checkout (move the file away, re-run, confirm it reappears or prints instructions correctly).
- [ ] Load once in a scratch Python session and confirm:
  ```python
  import pandas as pd
  df = pd.read_parquet("data/ridership_headline.parquet")
  print(df.shape)
  print(df.columns.tolist())
  print(df["date"].min(), df["date"].max())
  print(df["rail_mrt_kajang"].describe())
  ```
- [ ] Confirm the `rail_mrt_kajang` column exists and note the **date range** and **missing-value count**. Write these numbers down — you'll cite them in the README.

**Definition of done:** data downloads reproducibly, column exists, range and nulls recorded.
**Rubric:** contributes to *Reproducibility* (1 pt) — "dataset committed OR clear download instructions".

---

## Phase 2 — Exploratory Data Analysis (Notebook)

Work in `notebooks/01_eda_and_experiments.ipynb`.

### 2.1 Basic profiling
- [ ] Load the data, subset to `date` + `rail_mrt_kajang`, drop NaN target rows.
- [ ] Print `df.info()`, `df.describe()`, and the count of missing dates in the expected calendar range.
- [ ] Plot the **full time series** of daily ridership (line chart, x = date, y = trips).

### 2.2 Distribution & outliers
- [ ] Histogram and box plot of `rail_mrt_kajang`.
- [ ] Identify and note **outlier dates** (e.g. very low days — MCO-era, public holidays; very high days — free-ride promotions, major events).
- [ ] Compute IQR and flag days outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`.

### 2.3 Temporal structure
- [ ] Add `day_of_week`, `month`, `year`, `is_weekend` and plot **mean ridership by day-of-week**.
- [ ] Plot **monthly averages** to reveal seasonality (Ramadan, school holidays, year-end).
- [ ] Run a seasonal decomposition (`statsmodels.tsa.seasonal.seasonal_decompose`, period = 7) and plot trend/seasonal/residual.
- [ ] Compute and plot a **7-day rolling mean** to make the trend legible.
- [ ] Compute rolling std to see heteroskedasticity.

### 2.4 Structural breaks
- [ ] Mark the date of the **March 2025 peak-frequency change** (4 min → 3.7 min headway) on the time-series plot.
- [ ] Compare mean ridership in a window before vs. after that date. Record the delta — this is a headline EDA finding.

### 2.5 Relationship to candidate features
- [ ] Merge in a Malaysian **public-holiday calendar** (the `holidays` Python package supports `MY`) and compute mean ridership on holidays vs. non-holidays.
- [ ] Compute correlation of the target with lag features: `lag1`, `lag7`, `lag14`, `rolling_mean_7`, `rolling_mean_28`.
- [ ] Plot a **correlogram / ACF** of the series.
- [ ] (Optional) fetch a small weather dataset for KL and correlate rainfall/temperature with ridership — only if you have time; not required.

### 2.6 Feature-importance pre-check
- [ ] Fit a quick `RandomForestRegressor` on a rough feature set and plot `.feature_importances_`. This is your first pass at feature importance; the real one comes in Phase 4.

### 2.7 Write the EDA narrative
- [ ] Under each plot, write a **2–3 sentence interpretation** in markdown cells. This is what turns "basic EDA" (1 pt) into "extensive EDA" (2 pts) — the rubric rewards *analysis*, not just plots.

**Definition of done:** notebook runs top-to-bottom without errors; every plot has an interpretation; the 5–6 key findings are written down and will be quoted in the README.
**Rubric:** *EDA* → 2 pts.

---

## Phase 3 — Feature Engineering

Move reusable logic into `src/features.py` so both the notebook and `train.py` use the **same** code (this is what makes reproducibility real).

- [ ] Write `src/features.py` with a function `build_features(df: pd.DataFrame) -> pd.DataFrame` that adds:
  - [ ] Calendar: `day_of_week`, `is_weekend`, `month`, `day_of_year`, `is_month_start`, `is_month_end`.
  - [ ] Holiday: `is_holiday` (using `holidays` package, country = `MY`), plus maybe `days_to_next_holiday` / `days_from_prev_holiday`.
  - [ ] Lags: `lag_1`, `lag_7`, `lag_14`, `lag_28` on the target.
  - [ ] Rolling: `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7`.
  - [ ] (Optional) Fourier terms for day-of-year seasonality if you want a strong linear baseline.
- [ ] **Crucial:** drop rows with NaN created by lags *at the end of the build*, not before — otherwise you silently lose early data.
- [ ] Verify no **future leakage**: every feature at row `t` must be computable using only data up to `t − 1` (or calendar knowledge, which is fine). Lags must be `shift(+1)` not `shift(-1)`.
- [ ] Add a unit-test or assertion: for a sample of rows, assert that `lag_1` equals `target.shift(1)`.

**Definition of done:** `features.py` imports cleanly, notebook re-uses it, and a leakage spot-check passes.
**Rubric:** feeds into *Model training* (3 pts) and *Reproducibility* (1 pt).

---

## Phase 4 — Model Training & Experiments (Notebook)

Still in the notebook. Goal: decide the final model and its hyperparameters before you write `train.py`.

### 4.1 Time-based split
- [ ] Define a **cutoff date** (e.g. 80% of dates) and slice into `train`, `valid`, `test`. Print the date ranges of each.
- [ ] Do **not** use `train_test_split(shuffle=True)` anywhere. Use `TimeSeriesSplit` for CV if you need it.

### 4.2 Baseline
- [ ] Compute the **naive baseline**: predict `y[t] = y[t−7]` (same day last week). Record MAE/RMSE/MAPE.
- [ ] This number is your floor. Every model must beat it, and you must report it in the README.

### 4.3 Linear models
- [ ] Fit `Ridge` and `Lasso` on scaled features.
- [ ] Record validation MAE/RMSE/MAPE for each.

### 4.4 Tree-based models
- [ ] Fit `RandomForestRegressor`.
- [ ] Fit `GradientBoostingRegressor` **and** `XGBRegressor` (or LightGBM — pick one, XGBoost is fine).
- [ ] Record validation metrics for each.

### 4.5 Hyperparameter tuning
- [ ] Tune the **best tree model** with `GridSearchCV` or `RandomizedSearchCV` using `TimeSeriesSplit` as the CV splitter.
  - Candidates: `n_estimators`, `max_depth`, `min_samples_leaf`, `learning_rate`, `subsample`.
- [ ] Record the best params and the resulting validation metrics.

### 4.6 Model comparison
- [ ] Put all results in **one comparison table**: model | val MAE | val RMSE | val MAPE | tuned? | notes.
- [ ] Write a short **conclusion paragraph** picking the final model and justifying the choice.
- [ ] Also evaluate the chosen model **on the test set** — once, at the end. Report the test metrics. (Do not tune on test.)

### 4.7 Feature importance (final)
- [ ] For the chosen tree model, plot feature importances (or use `permutation_importance`).
- [ ] Interpret the top 5 features in markdown. Expect lags and `is_holiday`/`day_of_week` to dominate — if calendar features are near-zero importance, reconsider your feature set.

### 4.8 Save experiment artefacts
- [ ] Dump the comparison table to `notebooks/results.csv` (or just hard-code it into the README).
- [ ] Save the best hyperparameters to `notebooks/best_params.json` — `train.py` will read this.

**Definition of done:** comparison table exists, baseline beaten, chosen model justified, test metrics recorded.
**Rubric:** *Model training* → 3 pts (multiple model families **and** tuning).

---

## Phase 5 — Training Script (`src/train.py`)

Export the notebook logic to a script. This must be able to run **from a clean clone + downloaded data** and produce the model artefact.

- [ ] `train.py` structure:
  ```python
  # 1. load data
  # 2. build_features()
  # 3. time-based split
  # 4. read best_params.json
  # 5. fit final model on train+valid (or train only — pick one and document it)
  # 6. evaluate on test, print metrics
  # 7. joblib.dump(model, "models/model.joblib")
  # 8. write models/metrics.json
  ```
- [ ] Add `argparse` for `--data-path` and `--model-out` (nice-to-have, not required).
- [ ] Print test metrics at the end so the run is self-verifying.
- [ ] Run it end-to-end:
  ```bash
  python src/train.py
  ```
  - [ ] Confirm `models/model.joblib` is created.
  - [ ] Confirm `models/metrics.json` is written and contains test MAE/RMSE/MAPE.
- [ ] Delete `models/model.joblib`, re-run, confirm it is recreated — this is your reproducibility proof.

**Definition of done:** `python src/train.py` runs cleanly from a fresh venv and writes the model + metrics.
**Rubric:** *Exporting notebook to script* → 1 pt; *Reproducibility* → 1 pt.

---

## Phase 6 — Prediction Module (`src/predict.py`)

A thin module that loads `models/model.joblib` and exposes `predict(date: str) -> float`. The API will call this.

- [ ] Implement `load_model(path)` with `joblib.load`.
- [ ] Implement `predict_for_date(model, df, date)` that:
  - [ ] computes the lag/rolling features using the **last known actuals** from `df`,
  - [ ] returns a float.
- [ ] Handle the **"predict one day ahead" constraint** explicitly: state in a docstring that the function needs recent actuals to build lags. For a true multi-day-ahead forecast, the API would need to iterate.
- [ ] Quick smoke test:
  ```python
  from src.predict import load_model, predict_for_date
  model = load_model("models/model.joblib")
  print(predict_for_date(model, df, "2025-01-15"))
  ```

**Definition of done:** `predict_for_date` returns a sensible number for a date after the last actual.
**Rubric:** supports *Model deployment*.

---

## Phase 7 — FastAPI Service (`app/main.py`)

- [ ] Write `app/main.py`:
  ```python
  from fastapi import FastAPI
  from pydantic import BaseModel
  # load model once at startup
  # POST /predict  {date: "YYYY-MM-DD"} -> {date, predicted_ridership}
  # GET /health -> {"status": "ok"}
  ```
- [ ] Pydantic request/response models with **date validation** (reject malformed dates with 422).
- [ ] Run locally:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```
- [ ] Test with `curl`:
  ```bash
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"date": "2025-01-15"}'
  ```
- [ ] Test the **error paths**: missing `date`, bad format, date before training range. Confirm they return 4xx and not 500.
- [ ] Open `http://localhost:8000/docs` and confirm Swagger loads — nice screenshot material.

**Definition of done:** both endpoints respond correctly; error cases return 4xx.
**Rubric:** *Model deployment* → 1 pt.

---

## Phase 8 — Docker

- [ ] Write `Dockerfile`:
  - [ ] `FROM python:3.11-slim`
  - [ ] `WORKDIR /app`
  - [ ] `COPY requirements.txt .` then `RUN pip install --no-cache-dir -r requirements.txt`
  - [ ] `COPY app/ ./app/`, `COPY src/ ./src/`, `COPY models/ ./models/`
  - [ ] `EXPOSE 8000`
  - [ ] `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`
- [ ] Add `.dockerignore` (`.venv`, `data/`, `notebooks/`, `.git/`).
- [ ] Build:
  ```bash
  docker build -t mrt-kajang-forecast:latest .
  ```
- [ ] Run:
  ```bash
  docker run -d --name mrt-kajang-api -p 8000:8000 mrt-kajang-forecast:latest
  ```
- [ ] Verify the same `curl` tests work **against the container**, not just the venv.
- [ ] Confirm the model file was copied in (the container should not depend on the host filesystem).

**Definition of done:** container builds, starts, and `/predict` works from `curl`.
**Rubric:** *Containerization* → 2 pts.

---

## Phase 9 — Deployment on Proxmox

Choose **Option A (Compose)** or **Option B (K3s)**. Option A is faster and sufficient for the rubric; Option B is more impressive.

### Option A — Docker Compose on a Proxmox LXC/VM
- [ ] Create a small Debian/Ubuntu LXC (2 CPU, 2 GB RAM is plenty).
- [ ] Install Docker + Compose.
- [ ] Write `docker-compose.yml` with the `mrt-kajang-forecast` service, port mapping, and a volume mount for `models/`.
- [ ] `git clone` the repo on the LXC, `docker compose up -d`.
- [ ] From your laptop, hit the API at `http://<lxc-ip>:8000/predict` — this is the deployment proof.

### Option B — K3s on a Proxmox VM
- [ ] Create a Debian VM (2 CPU, 2 GB RAM).
- [ ] Install K3s: `curl -sfL https://get.k3s.io | sh -`.
- [ ] Write `k8s/deployment.yaml` (image, replicas=1, containerPort=8000) and `k8s/service.yaml` (NodePort 30080).
- [ ] `kubectl apply -f k8s/`.
- [ ] `kubectl get pods,svc` and confirm `Running` + NodePort assigned.
- [ ] Hit `http://<vm-ip>:30080/predict`.

### Evidence
- [ ] **Screenshot** of a successful `curl -X POST .../predict` with the JSON response visible.
- [ ] Save it to `docs/screenshot-deployed.png` and reference it in the README.
- [ ] Also capture the output of `docker compose ps` or `kubectl get pods`.

**Definition of done:** a remote host on Proxmox serves `/predict`; screenshot saved.
**Rubric:** *Cloud deployment* → 2 pts (code + URL/screenshot evidence).

---

## Phase 10 — README Polish

- [ ] Paste the README draft you already have.
- [ ] Fill in the **actual numbers** from your EDA and model runs:
  - [ ] dataset shape, date range, null count,
  - [ ] the comparison table from Phase 4.6,
  - [ ] final test MAE / RMSE / MAPE,
  - [ ] baseline metrics.
- [ ] Add the **screenshot** from Phase 9 with a caption.
- [ ] Add a **"Results"** section: one paragraph on which model won and why, plus a sentence on the lift over the naive baseline.
- [ ] Verify every command in the README actually works when copy-pasted.
- [ ] Verify attribution section is present (CC BY 4.0, DOSM, trips-not-persons note).

**Definition of done:** README is self-contained; a stranger can clone + follow it end-to-end.
**Rubric:** contributes across *Problem description* (2), *Reproducibility* (1), *Dependency management* (2), *Containerization* (2), *Cloud deployment* (2).

---

## Phase 11 — Final Reproducibility Test

Simulate a reviewer with a clean machine. Do this in a **fresh clone directory**:

- [ ] `git clone <repo> /tmp/review && cd /tmp/review`
- [ ] `python -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt` — succeeds?
- [ ] `python download_data.py` — gets the data?
- [ ] `python src/train.py` — trains and writes `models/model.joblib`?
- [ ] `uvicorn app.main:app --port 8000` — serves?
- [ ] `curl -X POST .../predict` — returns a prediction?
- [ ] `docker build . && docker run ...` — works?
- [ ] Any step that fails: fix it **in the repo**, not in your head. This is the step most people skip.

**Definition of done:** the entire pipeline runs on a clean clone with zero manual fixes.
**Rubric:** *Reproducibility* → 1 pt; protects *Dependency management* → 2 pts.

---

## Phase 12 — Self-Score Against Rubric

Tick each row only when you can point to a file or screenshot that proves it.

| Criterion | Max | Your score | Evidence file |
|---|---|---|---|
| Problem description | 2 | ☐ | `README.md` § Problem Description |
| EDA | 2 | ☐ | `notebooks/01_eda_and_experiments.ipynb` |
| Model training | 3 | ☐ | notebook § 4.3–4.6 + `notebooks/results.csv` |
| Export notebook → script | 1 | ☐ | `src/train.py` |
| Reproducibility | 1 | ☐ | `download_data.py` + README § Reproducibility |
| Model deployment | 1 | ☐ | `app/main.py` + `curl` output |
| Dependency management | 2 | ☐ | `requirements.txt` + README § Environment |
| Containerization | 2 | ☐ | `Dockerfile` + README § Containerisation |
| Cloud deployment | 2 | ☐ | `docker-compose.yml` / `k8s/` + `docs/screenshot-deployed.png` |
| **Total** | **16** | ☐ | |

Any row you can't tick: that's your remaining work.

---

## Common Pitfalls (read before you submit)

- **Data leakage via random split.** If your test MAE is suspiciously low (e.g. RMSE < 500 trips), you almost certainly shuffled time. Re-check Phase 4.1.
- **Lag leakage.** `shift(-1)` instead of `shift(+1)` gives you tomorrow's answer today. Re-check Phase 3.
- **Committing the Parquet file.** It's large; either commit it (rubric allows) or rely on `download_data.py` — but if you rely on the script, **test it on a clean clone** (Phase 11).
- **Dockerfile copies `data/`.** Don't — the container only needs `models/` and code.
- **README commands that don't work.** Every fenced block in the README must be runnable. Copy-paste each one.
- **No baseline.** Without the "same day last week" baseline, the reviewer can't tell if your model is any good.
- **Screenshots without context.** Crop to show the command *and* the JSON response.
- **Forgetting attribution.** CC BY 4.0 requires it. It's two lines; don't lose the point.

---

## Suggested Order of Effort (if time-constrained)

If you're short on time, this ordering maximises rubric points per hour:

1. Phase 1 → 2 → 4.1–4.6 (gets you EDA + modelling = 5 pts)
2. Phase 5 → 6 → 7 (script + API = 3 pts)
3. Phase 8 (Docker = 2 pts)
4. Phase 10 + 11 (README + clean-clone test = protects ~5 pts)
5. Phase 9 (Proxmox deploy = 2 pts — do this last, it's the most environment-dependent)
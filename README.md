

## Problem Description

The **MRT Kajang Line** is one of the busiest rail services in the Klang Valley, carrying hundreds of thousands of trips per day. During peak periods, the operator (Prasarana) adjusts train frequency and rolling-stock allocation to manage crowding. For example, in early 2025 the line moved from a 4-minute peak headway with 39 trains to a **3.7-minute headway with 42 trains** to alleviate congestion.
v  Malaysia
However, demand is not uniform across days. Public holidays, school terms, major events, and post-pandemic travel recovery all shift ridership up or down. A model that forecasts **next-day ridership** can help operations teams:

- anticipate high-demand days and pre-position spare trains,
- schedule maintenance on lower-demand days,
- compare actual demand against the service regime in place,
- and provide an early warning when ridership is likely to exceed a threshold that warrants additional crowd-management measures.

This project builds and deploys a **daily ridership forecasting model for the MRT Kajang Line**. The input is the recent history of ridership and calendar features; the output is a predicted trip count for a future date. The solution is served as a REST API so an operations dashboard or scheduling tool can request forecasts programmatically.

> **Granularity note.** The official MRT Kajang public dataset is **daily**, not hourly. Hourly station-to-station data is available for KTM Komuter, but not for MRT Kajang. This project therefore targets daily peak-demand forecasting. If hourly MRT data becomes available, the same pipeline can be adapted.

---

## Dataset

### Source

The primary dataset is **Daily Public Transport Ridership** from Malaysia’s open-data portal:

> `https://data.gov.my/data-catalogue/ridership_headline`

The dataset contains daily ridership totals for multiple services. The column used in this project is:

| Column | Type | Definition |
|---|---|---|
| `date` | Date (`YYYY-MM-DD`) | Calendar date |
| `rail_mrt_kajang` | Integer | MRT Kajang Line — **number of trips**, not unique individuals |

The column definition is explicit: each value is a **trip count**, not a count of distinct passengers. This is important for interpretation: one passenger making a return journey contributes two trips.

### License

The dataset is published under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)** , which permits commercial and non-commercial use, adaptation, and redistribution with attribution. This project includes the required attribution in the **Attribution** section below.

### Downloading the data

Download the **Full Dataset (Parquet)** from the dataset page above and place it at:

```
data/ridership_headline.parquet
```

Parquet is recommended over CSV because it preserves data types and is faster to load programmatically. The training script and notebook expect this path.

If you prefer to download via code, the dataset page provides a Parquet link. A minimal loader looks like this:

```python
import pandas as pd

df = pd.read_parquet("data/ridership_headline.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df[["date", "rail_mrt_kajang"]].dropna()
df = df.sort_values("date").reset_index(drop=True)
```

For reproducibility, the repository also includes `download_data.py`, which either downloads the file from the configured URL or prints instructions for manual download. The README does not assume that the data file is committed to Git.

---

## Project Structure

```
mrt-kajang-forecasting/
├── README.md
├── requirements.txt
├── Dockerfile
├── download_data.py
├── notebooks/
│   └── 01_eda_and_experiments.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── main.py
├── data/
│   └── .gitkeep
└── models/
    └── .gitkeep
```

---

## Exploratory Data Analysis

The notebook `notebooks/01_eda_and_experiments.ipynb` performs extensive EDA before modelling:

- **Range and distribution:** summary statistics of `rail_mrt_kajang`, histograms, and box plots.
- **Missing values:** count and calendar gaps; the dataset is daily but may have missing dates or null values.
- **Time-series structure:** daily, weekly, and monthly trends; day-of-week effects; holiday effects.
- **Target analysis:** seasonality decomposition, rolling means, and identification of structural breaks (for example, the March 2025 frequency change).
- **Feature importance analysis:** correlation between candidate calendar features and the target, and a preliminary tree-based feature-importance ranking.
- **Regime visualisation:** ridership before and after the 4-minute → 3.7-minute headway change.

The goal of EDA is not only to describe the data but to justify the feature set and the validation strategy used in the training script.

---

## Modelling

This is a **supervised regression** problem. The target is `rail_mrt_kajang`; the features are calendar and lag-based predictors constructed from the date and past ridership.

### Validation strategy

Because the data is temporal, random `train_test_split` would leak future information into the training set. The project uses a **time-based split**:

- training: all dates before a cutoff,
- validation: dates immediately after the cutoff,
- test: the final hold-out period.

A rolling-origin backtest is used for model comparison where appropriate.

### Models trained

At minimum, the project trains:

1. **Linear models** — Ridge and Lasso regression with calendar and lag features.
2. **Tree-based models** — Random Forest and Gradient Boosting (XGBoost or LightGBM).

Hyperparameters are tuned with **grid search or randomized search** using the time-based validation split. The tuning results and the final model choice are recorded in the notebook and in `src/train.py`.

### Evaluation metrics

- **MAE** (mean absolute error) — primary operational metric.
- **RMSE** — penalises large forecast errors.
- **MAPE** — useful for communicating error as a percentage of daily ridership.

A naive baseline (for example, “same day last week”) is included so that the model’s lift over a simple heuristic is explicit.

### Training script

`src/train.py` contains the production training logic:

```bash
python src/train.py
```

It loads the Parquet file, builds features, trains the selected model, evaluates it on the time-based test set, and writes the fitted model to `models/model.joblib`. The notebook is for exploration; the script is the reproducible training entry point.

---

## Deployment

The model is served with **FastAPI**. `app/main.py` exposes:

- `GET /health` — health check.
- `POST /predict` — accepts a JSON body with a date (and any required calendar context) and returns a predicted ridership value.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-11-10"}'
```

Example response:

```json
{"date": "2026-11-10", "predicted_ridership": 281430}
```

To run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Environment and Dependency Management

The project uses a **virtual environment**. The exact steps are:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

`requirements.txt` pins the core dependencies, including:

```
pandas
numpy
scikit-learn
xgboost
fastapi
uvicorn
joblib
pyarrow
```

The README gives the installation and activation instructions so that the environment is reproducible on another machine.

---

## Containerisation

A `Dockerfile` is provided at the project root. It installs dependencies, copies the application code, and starts the FastAPI service with Uvicorn.

Build the image:

```bash
docker build -t mrt-kajang-forecast:latest .
```

Run the container:

```bash
docker run -d \
  --name mrt-kajang-api \
  -p 8000:8000 \
  mrt-kajang-forecast:latest
```

Test it:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-11-10"}'
```

The Dockerfile is the single reproducible artefact for deployment; it removes “works on my machine” dependency drift.

---

## “Cloud” Deployment on Proxmox

The project is deployed on a **self-hosted Proxmox server**, which satisfies the cloud/kubernetes deployment criterion in the ML Zoomcamp rubric (local or remote Kubernetes is acceptable).

Two supported paths are documented:

### Option A — Docker Compose on a Proxmox VM/LXC

Create a small Linux VM or LXC container on Proxmox, install Docker, copy the repository, and run:

```bash
docker compose up -d
```

A `docker-compose.yml` file is included with the API service and a volume mount for the model artefact. This is the simplest production-like deployment on Proxmox.

### Option B — K3s on Proxmox (Kubernetes)

Install a lightweight Kubernetes distribution (K3s) on a Proxmox VM, then apply the provided manifests:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The service is exposed via a `NodePort` or an Ingress. This path demonstrates container orchestration and satisfies the higher end of the deployment rubric.

### Proof of deployment

After deployment, the API is reachable at the Proxmox host’s IP and port. The repository includes:

- a screenshot of a successful `curl` request to `/predict`,
- the exact `curl` command used,
- and the response body showing a predicted ridership value.

This is the “URL for testing — or video/screenshot of testing it” evidence required by the rubric.

---

## Reproducibility Checklist

To re-run the project from scratch:

1. Clone the repository.
2. Download `data/ridership_headline.parquet` from `https://data.gov.my/data-catalogue/ridership_headline` (Full Dataset → Parquet).
3. Create and activate a virtual environment.
4. `pip install -r requirements.txt`.
5. Run `python src/train.py` to reproduce the model.
6. Run `uvicorn app.main:app --host 0.0.0.0 --port 8000` to serve it.
7. Optionally build and run the Docker image.

The notebook can be executed independently for EDA, but the training script is the canonical, reproducible training path.

---

## Attribution

**Data source:** Department of Statistics Malaysia (DOSM) / Government of Malaysia, *Daily Public Transport Ridership*, `data.gov.my`. Licensed under **CC BY 4.0**. The dataset is derived from anonymous ticketing and tap-in/tap-out transactions of the public transport operators. Ridership values represent **trips**, not unique individuals.

Operational context on MRT Kajang peak-hour frequency is drawn from public statements by Prasarana Malaysia Berhad.
# 🌦️ Early Warning Framework for Extreme Weather — Lagos, Nigeria

An interactive Streamlit app built from the MSc thesis **"Design and Development of an
Early Warning Framework for Extreme Weather Conditions Using Ensemble Predictive
Analytics in Lagos State, Nigeria"** (Koleoso Busola Khadijat, TU2022000025, MSc
Computing, Anchor University Lagos, Supervisor: Dr D. Aleburu).

It combines a **live, real ensemble prediction demo** with a **dashboard of the
thesis's official Chapter 4 results**, and a **methodology walkthrough**.

## What's inside

| Page | What it does |
|---|---|
| `app.py` (Home) | Overview, the soft-voting formula, an honesty note on data scope |
| `pages/1_🌡️_Live_Prediction.py` | A real soft-voting ensemble (Logistic Regression + Random Forest + XGBoost), trained live on the sample dataset in `data/`, predicting tomorrow's heat/flood risk. Two modes: replay a real historical day, or explore a custom what-if weather scenario |
| `pages/2_📊_Results_Dashboard.py` | The thesis's official Chapter 4 evaluation (AUC-ROC, confusion matrices, feature importance, risk-tier breakdown), quoted from the thesis itself |
| `pages/3_📖_About.py` | Study area, data sources, feature engineering, model architecture, evaluation design, and the 3-tier alert system — from Chapters 1–3 |

### An important honesty note

This repository ships a **sample** of the thesis's full dataset (8 of Lagos's 20 LGAs,
1990–1999, ~29k daily records) — the subset that could be packaged with this demo. The
**Live Prediction** page trains a real ensemble on that sample, live, in your browser
session, and its own metrics are real but **modest** (particularly for flood, where this
smaller sample doesn't carry the same signal as the full study period). The **Results
Dashboard** page instead shows the thesis's actual official numbers, computed on the
complete engineered dataset over the full study period — quoted, not recomputed. The two
are deliberately kept separate throughout the app so nothing is overstated.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first load takes ~15–30 seconds while the
ensemble trains (cached after that, for the life of the process).

## Deploying to GitHub + Streamlit Community Cloud

### 1. Push this repo to GitHub

```bash
cd ews-lagos-streamlit
git init
git add .
git commit -m "Initial commit: Lagos Early Warning Framework demo app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

(Create the empty repository on GitHub first — [github.com/new](https://github.com/new) —
without a README/license so it doesn't conflict with this one.)

### 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub
   account.
2. Click **"New app"**.
3. Pick your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **"Deploy"**.

Streamlit Cloud installs `requirements.txt` automatically and starts the app — the first
deploy typically takes 2–5 minutes. Your app will be live at a URL like
`https://<your-app-name>.streamlit.app`.

No secrets or API keys are required — everything the app needs (the sample dataset) is
already in the `data/` folder.

### Updating the deployed app

Any push to the connected branch redeploys automatically:

```bash
git add .
git commit -m "Update app"
git push
```

## Project structure

```
ews-lagos-streamlit/
├── app.py                          # Home page
├── pages/
│   ├── 1_🌡️_Live_Prediction.py     # Live ensemble demo
│   ├── 2_📊_Results_Dashboard.py   # Official thesis results
│   └── 3_📖_About.py               # Methodology
├── src/
│   ├── model.py                    # Training + inference pipeline
│   ├── theme.py                    # Design system / CSS
│   └── thesis_results.py           # Official Chapter 4 figures (quoted)
├── data/
│   └── lagos_features.csv          # Sample feature dataset (8 LGAs, 1990–1999)
├── .streamlit/
│   └── config.toml                 # Theme colors
├── requirements.txt
└── README.md
```

## Methodology summary

Three base learners — Logistic Regression, Random Forest, and XGBoost — are trained
independently on ERA5 climate reanalysis data fused with WorldPop population, NBS
poverty (MPI), and ESA WorldCover/WUDAPT land-cover data, then combined by **soft
voting**: a plain average of their predicted probabilities.

```
P_ensemble(x) = (P_LR(x) + P_RF(x) + P_XGB(x)) / 3
```

Labels are shifted forward one day, so the model only ever predicts **tomorrow's**
extreme heat / flood risk using data known **up to and including today** — a genuine
early-warning design, not a same-day nowcast. See the in-app **About** page for the full
methodology.

## License / attribution

Built as a companion demo to an MSc thesis. Reuse for academic or educational purposes
is welcome with attribution to the author.

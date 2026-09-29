# 🌦️ Early Warning Framework for Extreme Weather — Lagos, Nigeria

An interactive Streamlit app built from the MSc thesis **"Design and Development of an
Early Warning Framework for Extreme Weather Conditions Using Ensemble Predictive
Analytics in Lagos State, Nigeria"** (Koleoso Busola Khadijat, TU2022000025, MSc
Computing, Anchor University Lagos, Supervisor: Dr D. Aleburu).

It combines a **live, real ensemble prediction demo** (with a prominent tomorrow's-alert
banner) with a **dashboard of the thesis's official Chapter 4 results**.

## What's inside

| Page | What it does |
|---|---|
| `app.py` (Home) | Overview, the soft-voting formula, the live prediction demo (real ensemble, trained on the sample dataset in `data/`) with a headline Low/Medium/High alert banner for tomorrow's heat and flood risk, and a condensed methodology summary |
| `pages/1_Results_Dashboard.py` | The thesis's official Chapter 4 evaluation (AUC-ROC, confusion matrices, feature importance, risk-tier breakdown), quoted from the thesis itself |

### An important honesty note

This repository ships a **sample** of the thesis's full dataset (8 of Lagos's 20 LGAs,
2010–2024, ~82k daily records) — a recent, sizeable slice chosen to keep real,
replayable historical days close to the present while staying under GitHub's file-size
warning threshold. The **live prediction demo** on the Home page trains a real ensemble
on that sample, live, in the app's session, and its own metrics are real but **not** the
thesis's official numbers. The **Results Dashboard** page instead shows the thesis's
actual official numbers, computed on the complete engineered dataset over the full
1990–2024 study period — quoted, not recomputed. The two are deliberately kept separate
throughout the app so nothing is overstated.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first load takes a little while as the
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

Streamlit Cloud installs `requirements.txt` automatically and starts the app. Your app
will be live at a URL like `https://<your-app-name>.streamlit.app`.

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
├── app.py                          # Home page: overview, live prediction + alert, methodology
├── pages/
│   └── 1_Results_Dashboard.py      # Official thesis Chapter 4 results
├── src/
│   ├── model.py                    # Training + inference pipeline
│   ├── theme.py                    # Design system / CSS
│   └── thesis_results.py           # Official Chapter 4 figures (quoted)
├── data/
│   └── lagos_features.csv          # Sample feature dataset (8 LGAs, 2010–2024)
├── .streamlit/
│   └── config.toml                 # Theme colors
├── runtime.txt                     # (currently unused by Streamlit Cloud's build image)
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
early-warning design, not a same-day nowcast. The predicted probability is converted
into a Low / Medium / High alert tier (thresholds 0.30 / 0.50) and shown as a headline
banner on the Home page. See the in-app methodology section for more detail.

## License / attribution

Built as a companion demo to an MSc thesis. Reuse for academic or educational purposes
is welcome with attribution to the author.

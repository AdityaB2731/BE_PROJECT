# Network Intrusion Detection System

This is a local-first Streamlit NIDS using scikit-learn. It does not depend on external raw dataset URLs.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Use the uploader in the sidebar, or add a CSV at `data/raw/CICIDS2017.csv`. The CSV must include a target column named `Label`, `Target`, `Class`, `Attack`, or `y`; `BENIGN` maps to 0 and all other labels map to 1. Numeric network-flow columns are used as features.

Training creates `artifacts/` locally with the Random Forest, Isolation Forest, scaler, and preprocessing metadata. These generated files are ignored by git.

## Project structure

```text
BE_PROJECT/
|-- app.py
|-- data_pipeline.py
|-- models.py
|-- risk_engine.py
|-- requirements.txt
|-- data/
|   |-- sample_nids.csv       # small test dataset
|   |-- raw/
|       |-- CICIDS2017.csv    # add your real dataset here later
|-- artifacts/                # generated after training
```

To test without downloading a dataset, enter `data/sample_nids.csv` in the Streamlit sidebar.
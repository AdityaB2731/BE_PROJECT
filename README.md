# Network Intrusion Detection System (NIDS)

A local-first Network Intrusion Detection System built with pure Python, scikit-learn, and Streamlit. The project analyzes CSV network-flow data, trains supervised and unsupervised detectors, and combines their results into a hybrid risk score.

The application does not download datasets from external raw URLs. You can upload a CSV through the Streamlit interface or provide a local file path.

## What This Project Detects

This implementation performs **binary intrusion detection**:

```text
BENIGN traffic -> 0 (normal)
Any other label -> 1 (attack)
```

It can recognize that traffic is suspicious, but it does not currently return the exact attack family as the prediction. For example, `DDoS`, `PortScan`, `Botnet`, and `Brute Force` are all treated as attack traffic during training.

The system is designed for CSV network-flow datasets, not raw packet captures. It does not currently parse `.pcap` files or perform multiclass attack classification.

## Main Features

- Streamlit dashboard with three tabs
- CSV drag-and-drop upload using `st.file_uploader()`
- Local CSV path fallback, useful for large datasets
- Automatic column-name whitespace cleanup
- Replacement of positive and negative infinity values
- Safe numeric conversion and median imputation
- Duplicate-row removal
- Case-insensitive `BENIGN` target mapping
- Stratified 80/20 train-test split
- StandardScaler persistence with joblib
- Supervised Random Forest classifier
- BENIGN-only Isolation Forest anomaly detector
- Hybrid weighted risk score
- LOW, MEDIUM, and HIGH severity levels
- Confusion matrix, classification report, and performance metrics
- Saved model artifacts for later flow inspection
- No Docker, Redis, Kafka, Kubernetes, or external service required

## Project Structure

```text
BE_PROJECT/
|-- app.py                         # Streamlit dashboard
|-- data_pipeline.py               # Loading, cleaning, encoding, splitting, scaling
|-- models.py                      # Random Forest, Isolation Forest, predictions, metrics
|-- risk_engine.py                 # Hybrid risk calculation and severity
|-- requirements.txt               # Python dependencies
|-- README.md
|-- .gitignore
|-- data/
|   |-- sample_nids.csv             # Small synthetic CICIDS-style test dataset
|   |-- raw/
|       |-- CICIDS2017.csv          # Place a real local dataset here later
|-- artifacts/                      # Generated after training; ignored by Git
|-- env/                            # Local virtual environment; ignored by Git
```

The sample file is intentionally small and is included for testing the application workflow. It uses CICIDS-style flow features and labels, but it is not an official CICIDS2017 capture.

## 1. Create the Virtual Environment

Open PowerShell in the project directory:

```powershell
cd "D:\BE PROJ RESEARCH\BE_PROJECT"
py -m venv env
```

Activate it:

```powershell
.\env\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The important packages are:

- `pandas` and `numpy` for data handling
- `scikit-learn` for both ML models and evaluation
- `joblib` for saving and loading models
- `streamlit` for the dashboard
- `matplotlib` and `seaborn` for the confusion matrix chart

## 2. Start the Dashboard

With the virtual environment active:

```powershell
streamlit run app.py
```

The browser opens the NIDS dashboard. The app does not require a dataset just to start. A dataset is required only when you click **Load and preview data** or **Train models**.

## 3. Choose a Dataset

There are three supported ways to provide data.

### Option A: Use the included sample

In the Streamlit sidebar, enter:

```text
data/sample_nids.csv
```

Then click **Load and preview data**, followed by **Train models**.

### Option B: Use a real local dataset

Place a CSV at:

```text
data/raw/CICIDS2017.csv
```

The application already uses that path as its default. You can also enter an absolute Windows path, for example:

```text
D:\datasets\CICIDS2017.csv
```

The local path option is recommended for large files because it avoids Streamlit uploader size limits.

### Option C: Upload through Streamlit

Use the sidebar uploader to drag and drop any compatible CSV file. Uploaded data is available to the current Streamlit session. For repeatable training, keep the dataset at a local path instead.

## 4. Dataset Requirements

The CSV must contain:

1. At least one target column.
2. At least one usable numeric feature column.
3. Both normal and attack rows.

The target column is detected automatically when its name is one of:

```text
Label
Target
Class
Attack
y
```

You can enter a different target-column name manually in the sidebar.

Example:

```csv
Flow Duration,Total Fwd Packets,Destination Port,Packet Length Mean,Label
1200,8,80,512.4,BENIGN
45,120,4444,75.2,DDoS
300,65,22,95.3,PortScan
```

The target labels do not need to be numeric. This is expected:

```text
BENIGN       -> 0
DDoS         -> 1
PortScan     -> 1
FTP-Patator  -> 1
Web Attack   -> 1
```

The feature columns should represent flow statistics such as duration, ports, packet counts, byte counts, packet lengths, and rates. Columns containing IP addresses, timestamps, or other nonnumeric values are converted to numeric values; columns that become entirely unusable are dropped.

## 5. What Happens During Preprocessing

When data is loaded, the pipeline performs these steps:

1. Reads the CSV using pandas.
2. Strips leading and trailing whitespace from every column name.
3. Replaces positive and negative infinity with missing values.
4. Removes duplicate rows.
5. Finds the target column.
6. Maps `BENIGN`, case-insensitively, to `0`.
7. Maps every other target value to `1`.
8. Removes the target from the feature table.
9. Converts feature columns to numeric values where possible.
10. Drops columns containing no usable numeric values.
11. Calculates median fill values for missing feature values.
12. Fills missing values and remaining empty numeric values with `0.0`.
13. Performs a stratified 80/20 train-test split.
14. Fits `StandardScaler` on the training set only.
15. Applies the fitted scaler to the test set.
16. Saves the scaler and preprocessing metadata for future inference.

Fitting the scaler only on training data prevents test-set information from leaking into training.

## 6. Supervised Model: Random Forest

The supervised detector is a `RandomForestClassifier` configured as follows:

```python
RandomForestClassifier(
	n_estimators=100,
	random_state=42,
	class_weight="balanced",
	n_jobs=-1,
)
```

### What supervised means here

Supervised learning uses rows that already have labels. The model learns a relationship between numeric flow features and the known target:

```text
flow features + known label -> learned classifier
```

During inference, the Random Forest returns the probability that a flow belongs to the attack class. This probability is used as the main component of the hybrid risk score.

### What it is good for

- Detecting patterns already represented in the labeled training data
- Producing an attack probability
- Evaluating accuracy, precision, recall, F1, ROC-AUC, and false-positive rate

### What it does not do

- It does not understand raw packets directly.
- It does not automatically identify a new attack family by name.
- It does not guarantee detection of attacks that are absent from the training data.
- It does not perform multiclass classification in the current implementation.

## 7. Unsupervised Model: Isolation Forest

The unsupervised detector is an `IsolationForest` trained strictly on BENIGN training rows:

```python
IsolationForest(
	n_estimators=100,
	contamination="auto",
	random_state=42,
)
```

### What unsupervised means here

The Isolation Forest does not use attack labels while it is fitting. It learns the normal traffic feature distribution:

```text
BENIGN flow features -> model of normal behavior
```

A flow that is easy to isolate from the normal distribution is treated as anomalous.

Scikit-learn returns:

```text
-1 -> anomaly
 1 -> normal
```

The project converts this to:

```text
1 -> anomaly
0 -> normal
```

It also derives a normalized anomaly score from the Isolation Forest decision function. This score is clipped to the range `0.0` to `1.0`, where higher values indicate more unusual behavior.

### What it is good for

- Finding traffic that differs from learned normal behavior
- Providing a second signal for unusual or previously unseen patterns
- Working with normal traffic examples without requiring attack labels during model fitting

### What it does not do

- It does not identify the attack name.
- It is not a replacement for a labeled classifier.
- An anomaly is not automatically proof of malicious activity; unusual legitimate traffic can also be anomalous.

## 8. What Is Hybrid in This Project?

The system is hybrid because it combines two different detection signals:

1. **Supervised signal:** Random Forest attack probability.
2. **Unsupervised signal:** Isolation Forest anomaly score.

The risk engine calculates:

```text
risk_score = (0.7 * supervised_attack_probability)
		   + (0.3 * anomaly_score)
```

The supervised model has the larger weight because it was trained directly on labeled normal and attack examples. The anomaly model contributes a secondary signal for traffic that looks unlike the BENIGN baseline.

Severity is assigned as follows:

```text
risk_score >= 0.70 -> HIGH
risk_score >= 0.40 -> MEDIUM
risk_score <  0.40 -> LOW
```

This means the hybrid engine is not a third ML model. It is a decision layer that combines the outputs of the supervised and unsupervised models into an operational risk classification.

## 9. Streamlit Dashboard

### Data & Model Training

This tab allows you to:

- Load a CSV from the uploader or local path
- Preview the first rows
- View row and column counts
- Train both models
- Save preprocessing and model artifacts

### Live Flow Inspector

After successful training, this tab creates numeric input fields for every feature used during training. It runs both models on the manually entered flow and displays:

- Risk score
- Supervised attack probability
- Anomaly score
- LOW, MEDIUM, or HIGH severity
- Isolation Forest status: NORMAL or ANOMALY

Manual input names and values must correspond to the feature columns from the training dataset. Missing values are filled using the saved training medians.

### Performance & Metrics

This tab displays:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- False-positive rate
- Confusion matrix
- Classification report

The current classification metrics evaluate the Random Forest on the held-out 20% test set. Isolation Forest does not receive a separate displayed performance report in the current dashboard.

## 10. Saved Artifacts

After training, the application creates the `artifacts/` directory:

```text
artifacts/
|-- random_forest.joblib       # trained supervised classifier
|-- isolation_forest.joblib    # trained BENIGN-only anomaly detector
|-- scaler.joblib              # fitted StandardScaler
|-- preprocessing.joblib       # feature names, medians, target metadata
```

These files let the application perform inference without retraining every time. They are generated locally and ignored by Git.

The prediction functions exposed by `models.py` are:

```python
predict_flow(features_dict)
predict_anomaly(features_dict)
```

`predict_flow()` returns the supervised attack probability. `predict_anomaly()` returns an anomaly status and anomaly score.

## 11. Benchmark Datasets

The included sample is only for testing the application. For meaningful evaluation, use a recognized network-intrusion benchmark dataset.

### CICIDS2017

Recommended first real dataset for this project. It contains BENIGN traffic and multiple attack categories with flow-based features.

- [CICIDS2017 official dataset page](https://www.unb.ca/cic/datasets/ids-2017.html)

### CSE-CIC-IDS2018

A larger and more recent CIC benchmark with multiple attack scenarios and network-flow data.

- [CSE-CIC-IDS2018 official dataset page](https://www.unb.ca/cic/datasets/ids-2018.html)

### UNSW-NB15

A widely used intrusion-detection benchmark containing normal traffic and several attack categories.

- [UNSW-NB15 official dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### Dataset usage notes

- Download datasets manually from their official pages or a trusted mirror.
- Extract the CSV files locally; the application does not load ZIP or PCAP files.
- Confirm that the CSV has both BENIGN and attack rows.
- Use the uploader for small files and the local path field for large files.
- Keep real datasets in `data/raw/`, which is ignored by Git.
- Do not commit private, sensitive, or licensed datasets to a public repository.

## 12. Git and Data Safety

The `.gitignore` excludes:

- `env/`, `.venv/`, and other virtual environments
- `artifacts/`
- Real datasets in `data/raw/`
- Environment secrets and Streamlit secrets
- Python caches, build output, coverage files, and editor metadata

The small `data/sample_nids.csv` file remains trackable so another developer can test the project immediately.

Never load untrusted `.joblib` files. Joblib uses Python object serialization and should only load artifacts generated by this project or obtained from a trusted source.

## 13. Troubleshooting

### Target column not found

Rename the target column to `Label`, or enter its exact name in the sidebar target-column field.

### No numeric feature columns found

The CSV contains only text or identifier fields. Use flow-statistic columns such as packet counts, byte counts, ports, durations, and packet lengths.

### Training requires both BENIGN and attack rows

The binary classifier requires both classes. The Isolation Forest also requires at least one BENIGN training row.

### Existing artifacts do not match the new dataset

Retrain after changing datasets. The saved feature list and scaler must match the dataset used for inference.

### Streamlit cannot upload a large file

Use the local path input instead of the uploader.

## License and Research Use

This project is intended for academic research, experimentation, and defensive security analysis. Model results depend on dataset quality, feature quality, class balance, and how closely future traffic resembles the training data. It should not be treated as the sole control for production security decisions.
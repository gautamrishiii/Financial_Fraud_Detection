# Financial Fraud Detection

A machine learning research project focused on detecting fraudulent financial transactions through multi-dataset analysis, feature engineering, temporal analysis, and model evaluation.

The project includes an interactive Streamlit dashboard for exploring datasets, assessing data quality, comparing model performance, and understanding fraud detection metrics.

## Project Objectives

- Analyze financial transaction datasets from different sources.
- Investigate fraud patterns and class imbalance.
- Perform data quality and temporal distribution analysis.
- Develop machine learning models for fraud classification.
- Evaluate models using fraud-focused performance metrics.
- Compare model performance across different datasets.
- Present findings through an interactive dashboard.

## Datasets

This project works with four financial fraud datasets:

| Dataset | Domain | Characteristics |
|---|---|---|
| MoMTSim V1 | Mobile Money | Synthetic transactions with fraud concentrated in transfers |
| MoMTSim V2 | Mobile Money | Synthetic dataset with high fraud prevalence |
| PaySim | Mobile Money | Simulated transactions with severe class imbalance and temporal changes |
| IEEE-CIS | Card Payments | Transaction and identity data with missing values and complex features |

### Dataset Statistics

| Dataset | Transactions | Fraud Cases | Fraud Rate |
|---|---:|---:|---:|
| MoMTSim V1 | 1,720,181 | 175,518 | 10.20% |
| MoMTSim V2 | 4,225,958 | 2,233,118 | 52.84% |
| PaySim | 6,362,620 | 8,213 | 0.129% |
| IEEE-CIS | 590,540 | 20,663 | 3.50% |

*Statistics reflect the dataset versions analyzed in this project.*

## Machine Learning

The project includes Logistic Regression-based fraud classification experiments.

Evaluation focuses on:

- PR-AUC (Precision-Recall Area Under the Curve)
- ROC-AUC (Receiver Operating Characteristic Area Under the Curve)
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Precision at 1% review capacity
- Recall at 1% review capacity

Special attention is given to class imbalance, temporal distribution changes, feature availability, and model generalization.

## Interactive Dashboard

The Streamlit application provides the following sections:

1. **Overview** — Dataset statistics and fraud prevalence visualizations.
2. **Datasets** — Dataset characteristics and comparisons.
3. **Data Quality** — Data quality issues and dataset limitations.
4. **Model Lab** — Model evaluation metrics and confusion matrices.
5. **Comparison** — Cross-dataset model performance comparisons.
6. **Reports** — Research reports and analysis outputs.
7. **Glossary** — Explanations of technical terminology.
8. **About** — Project information.
9. **User Management** — User and access management.

## Project Structure

```text
Financial_Fraud_Detection/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset/
│   └── raw/
│       ├── paysim/
│       ├── ieee_cis/
│       └── momtsim/
│
├── models/
│   ├── paysim/
│   └── ieee_cis/
│
├── reports/
│
├── src/
│
└── .streamlit/
    └── secrets.toml
```

*The structure above illustrates the main project components; actual files and folders may vary.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gautamrishiii/Financial_Fraud_Detection.git
cd Financial_Fraud_Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure authentication

Create `.streamlit/secrets.toml`:

```toml
[auth]
username = "admin"
password = "ChangeThisToAStrongPassword"
```

Replace the example password with a strong password. Do not commit this file to GitHub.

### 6. Run the dashboard

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- SQLite
- Git and GitHub

## Research Considerations

Financial fraud detection presents several challenges:

- Highly imbalanced transaction datasets.
- Changing fraud patterns over time.
- Differences between synthetic and real-world transaction data.
- Missing and unavailable features.
- Trade-offs between detecting fraud and generating false alarms.
- Generalization across different financial datasets.

The project emphasizes transparent evaluation rather than relying on accuracy alone.

## Limitations

- Synthetic datasets may not fully represent real financial behavior.
- Model performance can vary significantly across datasets.
- Evaluation results depend on the available features and experimental setup.
- The dashboard is intended for research and demonstration, not production financial transaction authorization.

## Future Improvements

- Additional machine learning algorithms.
- Advanced feature engineering.
- Temporal validation and out-of-time testing.
- Cross-dataset generalization experiments.
- Explainable AI and feature importance analysis.
- Improved fraud investigation workflows.

## Author

**Gautam Rishiii**

GitHub: [@gautamrishiii](https://github.com/gautamrishiii)

## License

License information will be added when the project license is selected.

by gautam rishi
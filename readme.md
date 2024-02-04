
# Credit Card Fraud Detection Evaluation

This project focuses on evaluating the performance of different machine learning models for credit card fraud detection using various sampling techniques. The goal is to handle imbalanced data and assess the models based on accuracy and recall metrics.

## Requirements

Make sure you have the following Python libraries installed:

```bash
pip install pandas imbalanced-learn scikit-learn xgboost
```

## Data

The dataset used for this project is assumed to be stored in a CSV file named `Creditcard_data.csv`. You can load the data using the following code:

```python
import pandas as pd

# Load data
credit_data = pd.read_csv('/path/to/Creditcard_data.csv')
```

## Sampling Techniques

The project utilizes the following sampling techniques from the `imbalanced-learn` library:

- Random Under-Sampling (RUS)
- Random Over-Sampling (ROS)
- Tomek Links (TL)
- Synthetic Minority Oversampling Technique (SMOTE)
- NearMiss (NM)

## Machine Learning Models

The project employs the following machine learning models:

- Logistic Regression (LR)
- Random Forest Classifier (RFC)
- Support Vector Classifier (SVC)
- K-Nearest Neighbors Classifier (KNC)
- XGBoost Classifier (XGB)

## Evaluation Metrics

The project evaluates model performance using two metrics:

1. Accuracy
2. Recall

## Usage

1. Ensure your dataset is loaded and named `credit_data`.
2. Modify the `models_list` variable to include the desired models.
3. Run the script to evaluate models based on both recall and accuracy metrics.

```python
python script_name.py
```

## Output

The script generates two result DataFrames:

1. `recall_results_dataframe`: Contains recall scores for each model and sampling technique.
2. `accuracy_results_dataframe`: Contains accuracy scores for each model and sampling technique.

The results are also saved to CSV files: `recall_results.csv` and `accuracy_results.csv`.


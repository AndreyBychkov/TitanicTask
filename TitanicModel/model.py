import os
import sys
import json
import pandas as pd
import logging

from collections import defaultdict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

log = logging.Logger("Model training logger")
log.setLevel(logging.DEBUG)

log_handler = logging.StreamHandler(sys.stdout)
log_handler.setLevel(logging.DEBUG)

log.addHandler(log_handler)


def train_model(train_data_file: str, output_dir: str):
    """
    Performs training of logistic regression and saves model.

    :param train_data_file: file with train data in CSV format.
    :param output_dir: folder, where results will be stored.
    :rtype: None
    """
    log.info("Start processing.")

    df = pd.read_csv(train_data_file)
    _preprocess_dataframe(df, output_dir)
    model = _train_log_regression_model(df)
    _save_model(model, df, output_dir)

    log.info("Model trained.")


def _preprocess_dataframe(df: pd.DataFrame, output_dir: str):
    log.info("Preprocessing DataFrame...")

    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    _preprocess_categorical_data(df, output_dir)
    _impute_data(df)


def _preprocess_categorical_data(df: pd.DataFrame, output_dir: str):
    """Imputes and encode categorical data."""
    for obj_col in df.select_dtypes('object'):
        df[obj_col] = df[obj_col].astype('category')

    for cat_col in df.select_dtypes('category'):
        df[cat_col].fillna(df[cat_col].mode().iloc[0], inplace=True)

    label_encoder_dict = defaultdict(LabelEncoder)

    for cat_col in df.select_dtypes('category'):
        df[cat_col] = label_encoder_dict[cat_col].fit_transform(df[cat_col])

    _save_label_encoders(label_encoder_dict, output_dir)


def _save_label_encoders(label_encoder_dict: defaultdict, output_dir: str):
    """Save encoders in output_dir/encoders folder"""
    for label, encoder in label_encoder_dict.items():
        with open(os.path.join(output_dir, 'encoders', f"{label}.json"), 'w') as f:
            json.dump(list(encoder.classes_), f)


def _impute_data(df: pd.DataFrame):
    for float_col in df.select_dtypes('float64'):
        df[float_col].fillna(df[float_col].mean(), inplace=True)

    for col in df.columns:
        df[col].fillna(df[col].mode().iloc[0], inplace=True)


def _train_log_regression_model(df: pd.DataFrame) -> LogisticRegressionCV:
    log.info("Training model...")

    X = df.drop(columns=['Survived'])
    y = df['Survived']
    calibrated_log_reg_model = LogisticRegressionCV(n_jobs=6)
    param_grid = {
        'penalty': ['l1'],
        'solver': ['liblinear'],  # one of the best for small datasets
        'tol': [1e-5, 1e-4, 1e-3],
        'max_iter': [150, 200, 250, 300],
        'intercept_scaling': [0.5, 1.0]
    }
    search = GridSearchCV(calibrated_log_reg_model, param_grid, cv=5)
    search.fit(X, y)
    return search.best_estimator_


def _save_model(model: LogisticRegressionCV, df: pd.DataFrame, output_dir: str):
    log.info("Saving model...")

    coefs = dict(zip(df.drop(columns=['Survived']).columns, model.coef_[0]))
    coefs['Intercept'] = model.intercept_[0]

    with open(os.path.join(output_dir, 'model_coefs.json'), 'w') as f:
        json.dump(coefs, f)


if __name__ == '__main__':
    train_path = r"../data/train.csv"
    output_path = r"../data/model"
    train_model(train_path, output_path)

# model_building.py  (enhanced)
#
# Key upgrades over the original:
#   • Tries three model families: LogisticRegression, LinearSVC, RandomForest
#   • 5-fold cross-validated GridSearchCV for each family
#   • Selects and saves the best model automatically
#   • Logs CV scores so you can compare across runs in MLflow
#   • Handles class imbalance with class_weight='balanced'

import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from src.logger import logging


# ---------------------------------------------------------------------------
# Candidate models and their hyperparameter grids
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = {
    'logistic_regression': {
        'model': LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        ),
        'param_grid': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
        },
    },
    'linear_svc': {
        # LinearSVC is very fast on TF-IDF vectors and often beats LR on text.
        # We wrap it in CalibratedClassifierCV to get predict_proba() for AUC.
        'model': CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=5000, random_state=42),
            cv=3,
        ),
        'param_grid': {
            'estimator__C': [0.01, 0.1, 1, 10],
        },
    },
    'random_forest': {
        'model': RandomForestClassifier(
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        ),
        'param_grid': {
            'n_estimators': [200, 400],
            'max_depth': [None, 30],
            'min_samples_leaf': [1, 3],
        },
    },
}


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s  (%d rows)', file_path, len(df))
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error loading data: %s', e)
        raise


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5,
) -> tuple:
    """
    Run GridSearchCV over all candidate models and return the best one.

    Returns:
        (best_model, best_name, best_score, all_results_dict)
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_score = -1
    best_model = None
    best_name  = ''
    all_results = {}

    for name, config in MODEL_CANDIDATES.items():
        logging.info('--- Tuning %s ---', name)
        gs = GridSearchCV(
            config['model'],
            config['param_grid'],
            cv=cv,
            scoring='roc_auc',          # AUC is more informative than accuracy
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        cv_score = gs.best_score_
        all_results[name] = {
            'best_params': gs.best_params_,
            'cv_auc':      round(cv_score, 4),
        }
        logging.info(
            '%s  best_params=%s  cv_AUC=%.4f',
            name, gs.best_params_, cv_score,
        )

        if cv_score > best_score:
            best_score = cv_score
            best_model = gs.best_estimator_
            best_name  = name

    logging.info(
        'Best model: %s  CV-AUC=%.4f', best_name, best_score,
    )
    return best_model, best_name, best_score, all_results


def save_model(model, file_path: str) -> None:
    """Persist the trained model to disk."""
    try:
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving model: %s', e)
        raise


def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        logging.info(
            'Training set: %d samples  |  class balance: %s',
            len(y_train),
            dict(zip(*np.unique(y_train, return_counts=True))),
        )

        best_model, best_name, best_score, all_results = train_best_model(X_train, y_train)

        # Print a summary table to stdout
        print('\n===== Model Selection Summary =====')
        for name, res in all_results.items():
            marker = '  <-- BEST' if name == best_name else ''
            print(f'  {name:25s}  cv_AUC={res["cv_auc"]:.4f}  params={res["best_params"]}{marker}')
        print(f'\nSaving {best_name} (CV-AUC={best_score:.4f})\n')

        save_model(best_model, 'models/model.pkl')

        # Also save selection metadata for evaluation step
        import json, os
        os.makedirs('reports', exist_ok=True)
        meta = {
            'best_model':  best_name,
            'cv_auc':      best_score,
            'all_results': all_results,
        }
        with open('reports/model_selection.json', 'w') as f:
            json.dump(meta, f, indent=4)
        logging.info('Model selection metadata saved to reports/model_selection.json')

    except Exception as e:
        logging.error('Failed to complete model building: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
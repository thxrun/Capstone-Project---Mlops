import os
import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

import mlflow
import mlflow.sklearn

from src.logger import logging


# -------------------------------------------------------------------------------------
# Production tracking setup — NO dagshub.init() (causes OAuth browser prompt in CI)
# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

repo_owner = "tharunkarthik2227"
repo_name  = "Capstone-Project---Mlops"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)


def load_model(file_path: str):
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error loading model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s (%d rows)", file_path, len(df))
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse CSV: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error loading data: %s", e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred       = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc       = roc_auc_score(y_test, y_pred_proba)
        mcc       = matthews_corrcoef(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics_dict = {
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1_score":  round(f1,        4),
            "auc":       round(auc,       4),
            "mcc":       round(mcc,       4),
            "tp": int(tp), "tn": int(tn),
            "fp": int(fp), "fn": int(fn),
        }

        logging.info("Evaluation metrics: %s", metrics_dict)

        print("\n===== Classification Report =====")
        print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
        print(f"AUC : {auc:.4f}")
        print(f"MCC : {mcc:.4f}")
        print(f"Confusion matrix — TP:{tp}  TN:{tn}  FP:{fp}  FN:{fn}")

        return metrics_dict

    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved to %s", file_path)
    except Exception as e:
        logging.error("Error saving metrics: %s", e)
        raise


def save_model_info(run_id: str, model_uri: str, file_path: str) -> None:
    try:
        with open(file_path, "w") as f:
            json.dump({"run_id": run_id, "model_uri": model_uri}, f, indent=4)
        logging.debug("Model info saved to %s", file_path)
    except Exception as e:
        logging.error("Error saving model info: %s", e)
        raise


def main():
    mlflow.set_experiment("my-dvc-pipeline")

    with mlflow.start_run() as run:
        try:
            clf       = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            os.makedirs("reports", exist_ok=True)
            save_metrics(metrics, "reports/metrics.json")

            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            if hasattr(clf, "get_params"):
                for k, v in clf.get_params().items():
                    mlflow.log_param(k, v)

            try:
                with open("reports/model_selection.json", "r") as f:
                    selection = json.load(f)
                mlflow.log_param("best_model_type", selection.get("best_model", "unknown"))
                mlflow.log_metric("cv_auc", selection.get("cv_auc", 0.0))
            except FileNotFoundError:
                logging.warning("model_selection.json not found — skipping selection metadata")

            logged_model = mlflow.sklearn.log_model(clf, name="model")

            save_model_info(
                run.info.run_id,
                logged_model.model_uri,
                "reports/experiment_info.json",
            )

            mlflow.log_artifact("reports/metrics.json")

        except Exception as e:
            logging.error("Failed to complete model evaluation: %s", e)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
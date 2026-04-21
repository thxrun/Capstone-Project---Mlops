import os
import json
import mlflow
import warnings
from mlflow import MlflowClient

from src.logger import logging

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------
# Production setup — NO dagshub.init() (causes OAuth browser prompt in CI)
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


def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            model_info = json.load(f)
        logging.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error loading model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    try:
        model_uri = model_info["model_uri"]
        logging.info("Registering model from URI: %s", model_uri)

        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        logging.info(
            "Model '%s' version %s registered successfully.",
            model_name,
            model_version.version,
        )

        # Transition the new version to Production
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,   # demote old Production versions
        )
        logging.info(
            "Model '%s' version %s transitioned to Production.",
            model_name,
            model_version.version,
        )

        return model_version

    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise


def main():
    try:
        model_info = load_model_info("reports/experiment_info.json")
        register_model("my_model", model_info)
    except Exception as e:
        logging.error("Failed to complete model registration: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
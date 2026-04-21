import os
import json
import mlflow
import dagshub
import warnings
from src.logger import logging

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------
# PRODUCTION SETUP
# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "tharunkarthik2227"
repo_name = "Capstone-Project---Mlops"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True,
)


# -------------------------------------------------------------------------------------
# CORE LOGIC
# -------------------------------------------------------------------------------------
def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    try:
        model_uri = model_info['model_uri']

        logging.info('Registering model from URI: %s', model_uri)
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logging.info(
            'Model %s version %s registered successfully.',
            model_name,
            model_version.version
        )

        return model_version

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)

    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
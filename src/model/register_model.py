import json
import mlflow
import dagshub
import warnings
from src.logger import logging

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for local use
mlflow.set_tracking_uri('https://dagshub.com/tharunkarthik2227/Capstone-Project---Mlops.mlflow')
dagshub.init(repo_owner='tharunkarthik2227', repo_name='Capstone-Project---Mlops', mlflow=True)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
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
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = model_info['model_uri']

        logging.info('Registering model from URI: %s', model_uri)
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

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
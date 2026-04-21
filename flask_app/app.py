from flask import Flask, render_template, request
import mlflow
import pickle
import os
import re
import string
import time
import warnings

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from prometheus_client import (
    Counter, Histogram, generate_latest,
    CollectorRegistry, CONTENT_TYPE_LATEST,
)

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

nltk.download('wordnet',   quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ---------------------------------------------------------------------------
# Negation words — must match training pipeline exactly
# ---------------------------------------------------------------------------
NEGATION_WORDS = {
    'no', 'not', 'nor', 'never', "n't", 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'barely', 'scarcely', 'without', 'against',
}

HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&#39;': "'", '&nbsp;': ' ', '&apos;': "'",
}

_STOP_WORDS  = set(stopwords.words('english')) - NEGATION_WORDS
_LEMMATIZER  = WordNetLemmatizer()


def _expand_contractions(text: str) -> str:
    contractions = {
        r"won\'t": "will not",    r"can\'t": "can not",
        r"couldn\'t": "could not", r"wouldn\'t": "would not",
        r"shouldn\'t": "should not", r"didn\'t": "did not",
        r"doesn\'t": "does not",  r"don\'t": "do not",
        r"isn\'t": "is not",      r"aren\'t": "are not",
        r"wasn\'t": "was not",    r"weren\'t": "were not",
        r"hasn\'t": "has not",    r"haven\'t": "have not",
        r"hadn\'t": "had not",    r"i\'m": "i am",
        r"i\'ve": "i have",       r"i\'ll": "i will",
        r"i\'d": "i would",       r"it\'s": "it is",
        r"that\'s": "that is",    r"there\'s": "there is",
        r"they\'re": "they are",  r"they\'ve": "they have",
        r"they\'ll": "they will", r"we\'re": "we are",
        r"we\'ve": "we have",     r"you\'re": "you are",
        r"you\'ve": "you have",   r"\'s": "",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = _expand_contractions(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.lower()
    text = re.sub(r'\b\d+\b', 'NUM', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in _STOP_WORDS]
    tokens = [_LEMMATIZER.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# MLflow / DagsHub — token-based auth only, NO dagshub.init()
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app",
    ["method", "endpoint"], registry=registry,
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds",
    ["endpoint"], registry=registry,
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class",
    ["prediction"], registry=registry,
)

# ---------------------------------------------------------------------------
# Load model and vectorizer
# ---------------------------------------------------------------------------
MODEL_NAME = "my_model"


def get_latest_model_version(model_name: str) -> str | None:
    client = mlflow.MlflowClient()
    for stage in ("Production", "None"):
        versions = client.get_latest_versions(model_name, stages=[stage])
        if versions:
            return versions[0].version
    return None


model_version = get_latest_model_version(MODEL_NAME)
if model_version is None:
    raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'")

model_uri = f"models:/{MODEL_NAME}/{model_version}"
print(f"Loading model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

vectorizer    = pickle.load(open("models/vectorizer.pkl", "rb"))
FEATURE_NAMES = vectorizer.get_feature_names_out().tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start    = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", result=None, error="Please enter some text.")

    cleaned      = normalize_text(text)
    features     = vectorizer.transform([cleaned])
    features_df  = pd.DataFrame(features.toarray(), columns=FEATURE_NAMES)
    result       = model.predict(features_df)
    prediction   = int(result[0])

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    return render_template("index.html", result=prediction)


@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
# app.py  (enhanced)
#
# Key fixes over the original:
#   1. normalize_text() now uses the SAME logic as the training pipeline
#      (negation preservation, contraction expansion, TF-IDF compatible) —
#      a mismatch here is a leading cause of production accuracy being lower
#      than eval accuracy.
#   2. Environment variable block consolidated — the original tried BOTH local
#      and production init, causing a crash when CAPSTONE_TEST is not set
#      during local development. Now a single conditional block handles both.
#   3. vectorizer column names built from the actual vocabulary, not just
#      range(n), so they always match what the model was trained on.
#   4. Minor: removed unused numpy import, fixed whitespace, added
#      a /healthz endpoint for Docker/k8s liveness probes.

from flask import Flask, render_template, request
import mlflow
import pickle
import os
import re
import string
import time
import warnings

import pandas as pd
import dagshub
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from prometheus_client import (
    Counter, Histogram, generate_latest,
    CollectorRegistry, CONTENT_TYPE_LATEST,
)

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Download required NLTK data (idempotent — safe to run on every startup)
# ---------------------------------------------------------------------------
nltk.download('wordnet',   quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ---------------------------------------------------------------------------
# Negation words that MUST NOT be stripped — same set as the training pipeline
# ---------------------------------------------------------------------------
NEGATION_WORDS = {
    'no', 'not', 'nor', 'never', "n't", 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'barely', 'scarcely', 'without', 'against',
}

HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&#39;': "'", '&nbsp;': ' ', '&apos;': "'",
}

# Build stop-word set once at module load — exclude negation words
_STOP_WORDS = set(stopwords.words('english')) - NEGATION_WORDS
_LEMMATIZER = WordNetLemmatizer()


def _expand_contractions(text: str) -> str:
    contractions = {
        r"won\'t": "will not",  r"can\'t": "can not",
        r"couldn\'t": "could not", r"wouldn\'t": "would not",
        r"shouldn\'t": "should not", r"didn\'t": "did not",
        r"doesn\'t": "does not", r"don\'t": "do not",
        r"isn\'t": "is not", r"aren\'t": "are not",
        r"wasn\'t": "was not", r"weren\'t": "were not",
        r"hasn\'t": "has not", r"haven\'t": "have not",
        r"hadn\'t": "had not", r"i\'m": "i am",
        r"i\'ve": "i have", r"i\'ll": "i will",
        r"i\'d": "i would", r"it\'s": "it is",
        r"that\'s": "that is", r"there\'s": "there is",
        r"they\'re": "they are", r"they\'ve": "they have",
        r"they\'ll": "they will", r"we\'re": "we are",
        r"we\'ve": "we have", r"you\'re": "you are",
        r"you\'ve": "you have", r"\'s": "",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_text(text: str) -> str:
    """
    Mirror of data_preprocessing.py's preprocess_text() used during training.
    Any deviation here causes train/serve skew and lower production accuracy.
    """
    if not isinstance(text, str):
        return ''

    # 1. HTML entities + tags
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Expand contractions BEFORE lowercasing
    text = _expand_contractions(text)

    # 3. Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # 4. Lowercase
    text = text.lower()

    # 5. Replace digits with NUM token (same as training)
    text = re.sub(r'\b\d+\b', 'NUM', text)

    # 6. Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)

    # 7. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Remove stop words (negation preserved)
    tokens = [w for w in text.split() if w not in _STOP_WORDS]

    # 9. Lemmatize
    tokens = [_LEMMATIZER.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# MLflow / DagsHub setup
# Checks for the production env-var; falls back to local dagshub.init().
# ---------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")

if dagshub_token:
    # Production: credentials come from the environment variable
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(
        "https://dagshub.com/tharunkarthik2227/Capstone-Project---Mlops.mlflow"
    )
else:
    # Local development: use dagshub.init() for interactive auth
    dagshub.init(
        repo_owner='tharunkarthik2227',
        repo_name='Capstone-Project---Mlops',
        mlflow=True,
    )
    mlflow.set_tracking_uri(
        'https://dagshub.com/tharunkarthik2227/Capstone-Project---Mlops.mlflow'
    )

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry,
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry,
)
PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry,
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

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Build feature column names from the vectorizer vocabulary — must match training
# TfidfVectorizer.get_feature_names_out() returns terms in vocabulary order
FEATURE_NAMES = vectorizer.get_feature_names_out().tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start = time.time()
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

    # Preprocess — must mirror training pipeline exactly
    cleaned = normalize_text(text)

    # Vectorize
    features = vectorizer.transform([cleaned])
    # Use vocabulary-derived column names (not plain integers)
    features_df = pd.DataFrame(features.toarray(), columns=FEATURE_NAMES)

    # Predict
    result = model.predict(features_df)
    prediction = int(result[0])

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    return render_template("index.html", result=prediction)


@app.route("/metrics")
def metrics():
    """Expose custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/healthz")
def healthz():
    """Liveness probe for Docker / Kubernetes."""
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
# preprocessing_utility.py  (enhanced)
#
# This utility is used for standalone testing and ad-hoc use.
# It mirrors data_preprocessing.py and app.py EXACTLY — any drift between
# these three files causes train/serve skew.

import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ---------------------------------------------------------------------------
# Negation words — MUST NOT be removed (critical for sentiment accuracy)
# ---------------------------------------------------------------------------
NEGATION_WORDS = {
    'no', 'not', 'nor', 'never', "n't", 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'barely', 'scarcely', 'without', 'against',
}

HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&#39;': "'", '&nbsp;': ' ', '&apos;': "'",
}

# Build once at import time
STOP_WORDS = set(stopwords.words('english')) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()


def _expand_contractions(text: str) -> str:
    """Expand common English contractions before any other processing."""
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


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline.  Must stay in sync with:
      • src/features/data_preprocessing.py
      • flask_app/app.py  → normalize_text()

    Steps (order matters):
        1. Decode HTML entities + strip tags
        2. Expand contractions
        3. Remove URLs / emails
        4. Lowercase
        5. Replace digits with NUM token
        6. Remove punctuation
        7. Collapse whitespace
        8. Remove stop words  (negation words preserved)
        9. Lemmatize
    """
    if not isinstance(text, str):
        return ''

    # 1. HTML
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Contractions
    text = _expand_contractions(text)

    # 3. URLs / emails
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # 4. Lowercase
    text = text.lower()

    # 5. Digit → NUM token
    text = re.sub(r'\b\d+\b', 'NUM', text)

    # 6. Punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)

    # 7. Whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Stop words
    tokens = [w for w in text.split() if w not in STOP_WORDS]

    # 9. Lemmatize
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


def remove_small_sentences(df: pd.DataFrame, column: str = 'text', min_words: int = 3) -> pd.DataFrame:
    """
    Drop rows where *column* contains fewer than *min_words* tokens after cleaning.

    Args:
        df        : Input DataFrame.
        column    : Text column name.
        min_words : Minimum token count to keep a row.

    Returns:
        Filtered DataFrame with reset index.
    """
    mask = df[column].apply(lambda x: len(str(x).split()) >= min_words)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick smoke test — run this file directly to verify the pipeline
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    test_cases = [
        ("This movie was absolutely amazing!", "positive"),
        ("I didn't like it at all. Not worth watching.", "negative — 'not' and 'didn't' preserved"),
        ("The film had no redeeming qualities.", "negative — 'no' preserved"),
        ("can't stop watching, won't stop recommending!", "positive — contractions expanded"),
        ("Visit https://example.com for more info. Score: 10/10", "URL + number handled"),
        ("<b>Great</b> acting &amp; story!", "HTML stripped"),
    ]

    print("=== Preprocessing utility smoke test ===\n")
    for raw, label in test_cases:
        processed = preprocess_text(raw)
        print(f"Input   : {raw}")
        print(f"Expected: {label}")
        print(f"Output  : {processed}")
        print()
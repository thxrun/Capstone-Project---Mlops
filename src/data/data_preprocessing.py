# data_preprocessing.py  (enhanced)

import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------------------------------------------------------------------------
# Negation words: removing these decimates sentiment accuracy.
# Keep them in the vocabulary by excluding them from the stopword list.
# ---------------------------------------------------------------------------
NEGATION_WORDS = {
    'no', 'not', 'nor', 'never', "n't", 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'barely', 'scarcely', 'without', 'against'
}

# HTML entity map for simple decoding without a heavy dependency
HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&#39;': "'", '&nbsp;': ' ', '&apos;': "'",
}


def _build_stopwords() -> set:
    """Return NLTK stopwords minus negation words (critical for sentiment)."""
    sw = set(stopwords.words('english')) - NEGATION_WORDS
    return sw


def _expand_contractions(text: str) -> str:
    """Expand common English contractions before any other processing."""
    contractions = {
        r"won\'t": "will not",
        r"can\'t": "can not",
        r"couldn\'t": "could not",
        r"wouldn\'t": "would not",
        r"shouldn\'t": "should not",
        r"didn\'t": "did not",
        r"doesn\'t": "does not",
        r"don\'t": "do not",
        r"isn\'t": "is not",
        r"aren\'t": "are not",
        r"wasn\'t": "was not",
        r"weren\'t": "were not",
        r"hasn\'t": "has not",
        r"haven\'t": "have not",
        r"hadn\'t": "had not",
        r"i\'m": "i am",
        r"i\'ve": "i have",
        r"i\'ll": "i will",
        r"i\'d": "i would",
        r"it\'s": "it is",
        r"that\'s": "that is",
        r"there\'s": "there is",
        r"they\'re": "they are",
        r"they\'ve": "they have",
        r"they\'ll": "they will",
        r"we\'re": "we are",
        r"we\'ve": "we have",
        r"you\'re": "you are",
        r"you\'ve": "you have",
        r"\'s": "",          # possessive strip
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Full preprocessing pipeline for a single review string.
    Order matters — steps are arranged to avoid information loss.
    """
    if not isinstance(text, str):
        return ''

    # 1. Decode HTML entities and strip tags
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    text = re.sub(r'<[^>]+>', ' ', text)          # remove HTML tags

    # 2. Expand contractions BEFORE lowercasing (regex is case-insensitive)
    text = _expand_contractions(text)

    # 3. Remove URLs and email addresses
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # 4. Lowercase
    text = text.lower()

    # 5. Replace numbers — don't remove outright; replace with a token
    #    so the model can learn patterns like "10/10" or "0 stars"
    text = re.sub(r'\b\d+\b', 'NUM', text)

    # 6. Remove punctuation (but keep apostrophes handled above)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)

    # 7. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Remove stopwords (negation words are preserved)
    tokens = [w for w in text.split() if w not in stop_words]

    # 9. Lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame, col: str = 'review') -> pd.DataFrame:
    """
    Apply text preprocessing to *col* in-place and drop short/empty rows.

    Args:
        df  : Input DataFrame.
        col : Name of the text column.

    Returns:
        Cleaned DataFrame.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = _build_stopwords()

    df = df.copy()
    df[col] = df[col].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

    # Drop rows that became too short after cleaning (< 3 tokens)
    mask = df[col].apply(lambda x: len(str(x).split()) >= 3)
    dropped = (~mask).sum()
    if dropped:
        logging.info('Dropped %d rows with < 3 tokens after preprocessing', dropped)
    df = df[mask].reset_index(drop=True)

    logging.info('Data preprocessing completed — %d rows remain', len(df))
    return df


def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data  = pd.read_csv('./data/raw/test.csv')
        logging.info('Raw data loaded: train=%d  test=%d', len(train_data), len(test_data))

        train_processed = preprocess_dataframe(train_data, 'review')
        test_processed  = preprocess_dataframe(test_data,  'review')

        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(data_path,  'test_processed.csv'),  index=False)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete data preprocessing: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
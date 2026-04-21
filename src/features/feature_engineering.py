# feature_engineering.py  (enhanced)
#
# Key upgrades over the original:
#   • TfidfVectorizer instead of CountVectorizer
#     – TF-IDF down-weights common words that appear in nearly every review
#       (e.g. "movie", "film") and up-weights discriminative words.
#   • unigrams + bigrams (ngram_range=(1,2))
#     – "not good" and "not bad" are each a single bigram feature,
#       not two separate words. This dramatically helps negation handling.
#   • sublinear_tf=True  — log-scaling of term frequencies reduces the
#     impact of a word appearing 50 times vs 5 times.
#   • min_df=3  — prune extremely rare terms that only cause over-fitting.

import numpy as np
import pandas as pd
import os
import pickle
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV and fill NaN text values with empty strings."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded from %s  (%d rows)', file_path, len(df))
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse CSV: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error loading data: %s', e)
        raise


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
    ngram_max: int = 2,
    min_df: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a TF-IDF vectoriser on training text and transform both splits.

    Args:
        train_data   : Training DataFrame with 'review' and 'sentiment' columns.
        test_data    : Test DataFrame with the same columns.
        max_features : Vocabulary size cap (read from params.yaml).
        ngram_max    : Upper end of n-gram range.  Default=2 (unigrams+bigrams).
        min_df       : Minimum document frequency for a term to be kept.

    Returns:
        (train_df, test_df) as dense DataFrames with a trailing 'label' column.
    """
    try:
        logging.info(
            'Applying TF-IDF: max_features=%d  ngram_range=(1,%d)  min_df=%d',
            max_features, ngram_max, min_df,
        )

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            sublinear_tf=True,          # log(1 + tf) — reduces impact of high-freq terms
            min_df=min_df,              # ignore terms seen in fewer than min_df docs
            strip_accents='unicode',
            analyzer='word',
        )

        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test  = test_data['review'].values
        y_test  = test_data['sentiment'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf  = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        os.makedirs('models', exist_ok=True)
        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
        logging.info(
            'TF-IDF applied. Vocab size: %d  Train shape: %s  Test shape: %s',
            len(vectorizer.vocabulary_), train_df.shape, test_df.shape,
        )

        return train_df, test_df

    except Exception as e:
        logging.error('Error during TF-IDF transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save a DataFrame to CSV."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error saving data: %s', e)
        raise


def main():
    try:
        params       = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        ngram_max    = params['feature_engineering'].get('ngram_max', 2)
        min_df       = params['feature_engineering'].get('min_df', 3)

        train_data = load_data('./data/interim/train_processed.csv')
        test_data  = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features, ngram_max, min_df)

        save_data(train_df, os.path.join('./data', 'processed', 'train_bow.csv'))
        save_data(test_df,  os.path.join('./data', 'processed', 'test_bow.csv'))

    except Exception as e:
        logging.error('Failed to complete feature engineering: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
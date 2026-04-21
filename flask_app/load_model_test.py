# load_model_test.py  (enhanced)
#
# Run this from the project root to verify:
#   1. The vectorizer loads and has a sensible vocabulary
#   2. The preprocessing pipeline produces output consistent with what the
#      vectorizer was trained on
#   3. End-to-end: text → preprocess → vectorize → shape check

import os
import pickle
import sys

# Allow running from project root OR from tests/ / flask_app/ subdirectory
SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl'),
    os.path.join(os.path.dirname(__file__), 'models', 'vectorizer.pkl'),
    '../models/vectorizer.pkl',
    'models/vectorizer.pkl',
]

def find_vectorizer() -> str:
    for path in SEARCH_PATHS:
        if os.path.exists(path):
            return path
    return None


def main():
    # -------------------------------------------------------------------
    # 1. Locate and load the vectorizer
    # -------------------------------------------------------------------
    pkl_path = find_vectorizer()
    if pkl_path is None:
        print("ERROR: vectorizer.pkl not found. Checked paths:")
        for p in SEARCH_PATHS:
            print(f"  {p}")
        sys.exit(1)

    print(f"Found vectorizer at: {pkl_path}")
    try:
        vectorizer = pickle.load(open(pkl_path, 'rb'))
        print("Vectorizer loaded successfully.")
    except Exception as e:
        print(f"ERROR loading vectorizer: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------
    # 2. Inspect vocabulary
    # -------------------------------------------------------------------
    vocab_size = len(vectorizer.vocabulary_)
    print(f"\nVocabulary size : {vocab_size}")
    print(f"Vectorizer type : {type(vectorizer).__name__}")

    # Check that bigrams are present (expected after the enhancement)
    bigram_examples = [t for t in vectorizer.vocabulary_ if ' ' in t][:5]
    if bigram_examples:
        print(f"Bigram examples : {bigram_examples}  ✓")
    else:
        print("WARNING: No bigrams found — did you use ngram_range=(1,2) in feature_engineering.py?")

    # Check that negation terms survived
    neg_terms_in_vocab = [t for t in ['not', 'no', 'never', 'not good', 'not bad'] if t in vectorizer.vocabulary_]
    if neg_terms_in_vocab:
        print(f"Negation terms  : {neg_terms_in_vocab}  ✓")
    else:
        print("WARNING: Negation terms missing from vocabulary — check stop-word handling.")

    # -------------------------------------------------------------------
    # 3. End-to-end smoke test: preprocess → vectorize
    # -------------------------------------------------------------------
    # Import the shared preprocessing function
    try:
        # Try relative import (project installed as a package)
        from preprocessing_utility import preprocess_text
    except ImportError:
        # Fallback: add parent directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        from preprocessing_utility import preprocess_text

    test_reviews = [
        "This film was absolutely not boring. I loved every minute!",
        "Terrible movie. I didn't enjoy it at all — waste of time.",
        "The visuals were stunning but the story had no depth.",
    ]

    print("\n=== End-to-end smoke test ===")
    for review in test_reviews:
        cleaned = preprocess_text(review)
        features = vectorizer.transform([cleaned])
        print(f"\nRaw     : {review}")
        print(f"Cleaned : {cleaned}")
        print(f"Shape   : {features.shape}  non-zero features: {features.nnz}")

    print("\nAll checks passed.")


if __name__ == '__main__':
    main()
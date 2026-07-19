from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import os
import config

def vectorize_texts(X_train, X_test, save=True):
    vectorizer_path = os.path.join(config.FEATURES_DIR, "tfidf_vectorizer.joblib")
    if os.path.exists(vectorizer_path):
        print(f"Vectorizer already exists {vectorizer_path}")
        vectorizer = load(vectorizer_path)
        X_train_vec = vectorizer.transform(X_train).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()

        return X_train_vec, X_test_vec, vectorizer
    else:
        vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=config.NGRAM_RANGE,
            sublinear_tf=True
        )
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()

        if save and config.FEATURES_DIR:
            os.makedirs(config.FEATURES_DIR, exist_ok=True)
            dump(vectorizer, vectorizer_path)
            print(f"Vectorizer saved to {vectorizer_path}")

        return X_train_vec, X_test_vec, vectorizer
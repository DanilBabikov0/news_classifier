from sklearn.feature_extraction.text import TfidfVectorizer
import config

def vectorize_texts(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    return X_train_vec, X_test_vec, vectorizer
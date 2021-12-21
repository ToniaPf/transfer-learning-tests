import os
import random

import numpy as np
import svmlight
from gensim.corpora import SvmLightCorpus
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import fetch_20newsgroups, dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    removed = []
    for r in words:
        if r not in stop_words:
            removed.append(r)
    result = ' '.join(removed)
    return result


def get_svmlight_data(svm_file):
    corpus = SvmLightCorpus(svm_file)
    lines = list(corpus)
    labels = corpus.labels
    labels = [-1 if label == "-1" else +1 for label in labels]
    return list(zip(labels, lines))


class SVMLight(BaseEstimator, ClassifierMixin):
    def __init__(self, **train_params):
        self.train_params = train_params
        self._random_seed = random.randint(1000, 9999)
        self.model = None

    def fit(self, X, y):
        _y = np.array([1 if yy == 1 else -1 for yy in y.ravel()])
        dump_svmlight_file(X, _y, f'train_data_{self._random_seed}', zero_based=False)
        train_data = get_svmlight_data(f'train_data_{self._random_seed}')
        os.remove(f'train_data_{self._random_seed}')
        self.model = svmlight.learn(train_data, **self.train_params)

    def predict(self, X):
        predictions = self.predict_proba(X)
        return predictions.argmax(1)

    def predict_proba(self, X):
        dump_svmlight_file(X, [0] * X.shape[0], f'test_data{self._random_seed}', zero_based=False)
        test_data = get_svmlight_data(f'test_data{self._random_seed}')
        os.remove(f'test_data{self._random_seed}')
        predictions = []
        for value in svmlight.classify(self.model, test_data):
            if value <= -1:
                predictions.append([1, 0])
            elif value >= 1:
                predictions.append([0, 1])
            else:
                v = (value + 1) / 2
                predictions.append([1-v, v])
        return np.array(predictions)


if __name__ == "__main__":
    source = fetch_20newsgroups(
        categories=[
            'sci.crypt',
            'sci.electronics',
            # 'sci.med',
            # 'sci.space',
            'talk.politics.guns',
            'talk.politics.mideast'
            # 'talk.politics.misc',
            # 'talk.religion.misc',
        ],
        remove=('headers', 'footers', 'quotes')
    )

    target = fetch_20newsgroups(
        categories=[
            # 'sci.crypt',
            # 'sci.electronics',
            'sci.med',
            'sci.space',
            # 'talk.politics.guns',
            # 'talk.politics.mideast'
            'talk.politics.misc',
            'talk.religion.misc',
        ],
        remove=('headers', 'footers', 'quotes')
    )

    x_source = np.array([remove_stopwords(t) for t in source.data])
    y_source = np.array([1 if t > 1 else 0 for t in source.target])
    y_source = y_source[x_source != ""]
    x_source = x_source[x_source != ""]

    x_target = np.array([remove_stopwords(t) for t in target.data])
    y_target = np.array([1 if t > 1 else 0 for t in target.target])
    y_target = y_target[x_target != ""]
    x_target = x_target[x_target != ""]

    vectorizer = TfidfVectorizer(max_features=10_000)
    vectorizer.fit(np.concatenate((x_source, x_target)))
    x_source_v = vectorizer.transform(x_source).toarray()
    x_target_v = vectorizer.transform(x_target).toarray()

    svm = SVMLight(type="classification", kernel="linear")
    svm.fit(x_source_v, y_source)
    pred = svm.predict(x_target_v)
    print(accuracy_score(y_target, pred))

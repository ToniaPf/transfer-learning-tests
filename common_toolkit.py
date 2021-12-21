import copy
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from toolkit import TrAdaBoost
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


def run_experiment_toolkit(
        x_source, y_source,
        x_target, y_target,
        x_target_test, y_target_test,
        estimators=[],
        ratio=1,
        csv_file='',
        tfidf=True,
        ):

    with open(csv_file, 'a') as f:

        writer = csv.writer(f)

        K = 3
        train_size = round(x_source.shape[0] * ratio)
        if train_size >= x_target.shape[0]:
            train_size = x_target.shape[0] - 2

        sss = StratifiedShuffleSplit(n_splits=K, train_size=train_size, random_state=42)

        for i, (train_index, test_index) in enumerate(sss.split(x_target, y_target)):

            print(f'kFold: {i}')
            x_target_train = x_target[train_index]
            y_target_train = y_target[train_index]

            if tfidf:
                all_raw = np.concatenate((x_source, x_target_train), axis=0)
                vectorizer = TfidfVectorizer(lowercase=True, max_features=5_000)
                vectorizer.fit(all_raw)

                x_source_v = vectorizer.transform(x_source).toarray()
                x_target_train_v = vectorizer.transform(x_target_train).toarray()
                x_target_test_v = vectorizer.transform(x_target_test).toarray()
            else:
                x_source_v = x_source
                x_target_train_v = x_target_train
                x_target_test_v = x_target_test

            for (name, base_estimator) in estimators:

                ## Test1 --> Train Source // Test target ###
                est = copy.copy(base_estimator)
                est.fit(x_source_v, y_source)
                preds1 = est.predict(x_target_test_v)
                writer.writerow([f'{name}', str(ratio), 'test1', 100 * accuracy_score(y_target_test, preds1)])

                ## Test2 --> Train source+target --> Test target ###
                est = copy.copy(base_estimator)
                est.fit(x_source_v, y_source)
                both_x_train = np.concatenate((x_target_train_v, x_source_v), axis=0)
                both_y_train = np.concatenate((y_target_train, y_source), axis=0)
                est.fit(both_x_train, both_y_train)
                preds2 = est.predict(x_target_test_v)
                writer.writerow([f'{name}', str(ratio), 'test2', 100 * accuracy_score(y_target_test, preds2)])

                ## Test3 --> Train target --> Test target ###
                est = copy.copy(base_estimator)
                est.fit(x_target_train_v, y_target_train)
                preds3 = est.predict(x_target_test_v)
                writer.writerow([f'{name}', str(ratio), 'test3', 100 * accuracy_score(y_target_test, preds3)])

                class MyTrAdaBoost(TrAdaBoost):
                    def train_classify(self, trans_data, trans_label, test_data, P):
                        clf = copy.copy(base_estimator)
                        clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
                        return clf.predict(test_data)

                n_iters = min(50, x_target_train_v.shape[0])
                tr = MyTrAdaBoost(n_iters)
                preds = tr.fit_predict(x_source_v, x_target_train_v, y_source, y_target_train, x_target_test_v)
                writer.writerow([f'{name}', str(ratio), 'TrAdaBoost', 100 * accuracy_score(y_target_test, preds)])


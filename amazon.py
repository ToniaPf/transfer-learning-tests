import copy
import os
import csv
from datetime import datetime

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

import numpy as np
from nltk.corpus import stopwords
# from common_toolkit import run_experiment_toolkit
# data input
from common_adapt import run_experiment_adapt
# from svm_light_transfer import SVMLight
from common_toolkit import run_experiment_toolkit


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    removed = []
    for r in words:
        if r not in stop_words:
            removed.append(r)
    result = ' '.join(removed)
    return result


csv_file = f'amazon_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'


with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Estimator', 'Ratio', 'Test-Type', 'Accuracy'])

dir_path = os.path.dirname(os.path.realpath(__file__))

df_source = pd.read_csv(os.path.join(dir_path, "../data", "amazon_reviews_us_Kitchen_v1_00.tsv"),
                        sep="\t",
                        error_bad_lines=False,
                        usecols=['star_rating',
                                 'review_headline',
                                 'review_body'],
                        nrows=100000,
                        # skiprows=lambda i: i>0 and random.random() > p
                        )

df_target = pd.read_csv(os.path.join(dir_path, "../data", "amazon_reviews_us_Toys_v1_00.tsv"),
                        sep="\t",
                        error_bad_lines=False,
                        usecols=['star_rating',
                                 'review_headline',
                                 'review_body'],
                        nrows=100000,
                        # skiprows=lambda i: i>0 and random.random() > p
                        )

df_source['merged_review'] = df_source['review_headline'] + " " + df_source['review_body']
df_target['merged_review'] = df_target['review_headline'] + " " + df_target['review_body']
df_source.drop(['review_headline', 'review_body'], inplace=True, axis=1)
df_target.drop(['review_headline', 'review_body'], inplace=True, axis=1)
df_source.dropna(inplace=True)
df_target.dropna(inplace=True)

source_size = 8_000
df_source = df_source.groupby("star_rating").sample(int(source_size/4))


df_source['rating'] = [0 if x < 3 else 1 for x in df_source['star_rating']]
df_target['rating'] = [0 if x < 3 else 1 for x in df_target['star_rating']]

df_source.drop(df_source[df_source.star_rating == 3].index, inplace=True)
df_target.drop(df_target[df_target.star_rating == 3].index, inplace=True)

df_source['merged_review'] = df_source['merged_review'].apply(remove_stopwords).str.lower()
df_target['merged_review'] = df_target['merged_review'].apply(remove_stopwords).str.lower()

df_target_train = df_target.groupby("star_rating").head(source_size * 0.5 / 4)
df_target_test = df_target.groupby("star_rating").tail(source_size * 0.5 / 4)

df_source.drop('star_rating', inplace=True, axis=1)
df_target.drop('star_rating', inplace=True, axis=1)
df_target_test.drop('star_rating', inplace=True, axis=1)

x_source = df_source['merged_review'].to_numpy()
y_source = np.array(df_source['rating'])

x_target = df_target_train['merged_review'].to_numpy()
y_target = np.array(df_target_train['rating'])

x_target_test = df_target_test['merged_review'].to_numpy()
y_target_test = np.array(df_target_test['rating'])


estimators = [
    ('dtreegini', DecisionTreeClassifier(max_features="log2", splitter="random", criterion="gini")),
    # ('dtree5', DecisionTreeClassifier(max_depth=5, random_state=1),),
    # ('dtree10', DecisionTreeClassifier(max_depth=10, random_state=1)),
    # ('gnb', GaussianNB()),
    # ('lsvc', NaivelyCalibratedLinearSVC()),
    # ('lr', LogisticRegression()),
    # ('svmlight', SVMLight(type="classification", kernel="linear"))
]


run_msda = False  # both
run_tradaboost = False  # both
kmm_kernels = ['linear', 'rbf', 'poly']  # adapt
run_fe = False  # adapt
run_coral = False  # adapt
run_regular_transfer = False  # adapt
run_kmm = False

for ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    print(f"Ratio : {ratio}")

    run_experiment_adapt(
        x_source, y_source,
        x_target, y_target,
        x_target_test, y_target_test,
        estimators,
        run_tradaboost,
        run_kmm,
        kmm_kernels,
        run_fe,
        run_coral,
        run_msda,
        run_regular_transfer,
        ratio,
        csv_file
    )

    print('Toolkit-Amazon')
    run_experiment_toolkit(
        x_source, y_source,
        x_target, y_target,
        x_target_test, y_target_test,
        estimators,
        ratio,
        csv_file
    )

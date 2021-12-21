import csv
from datetime import datetime
from scipy.io import arff

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer , CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords

# data input
from common_adapt import run_experiment_adapt
# from common_toolkit import run_experiment_toolkit
# from svm_light_transfer import SVMLight
from common_toolkit import run_experiment_toolkit

csv_file = f'reuters_adapt_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'

with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Estimator', 'Ratio', 'Test-Type', 'Accuracy'])


source = arff.loadarff(r"..\reuters\PeoplePlaces.src.arff")
target = arff.loadarff(r"..\reuters\PeoplePlaces.tar.arff")

df_source = pd.DataFrame(source[0])
df_target = pd.DataFrame(target[0])

df_target_train = df_target.groupby('class').head(250)
df_target_test = df_target.groupby('class').tail(200)


x_source = df_source.drop(df_source.columns[len(df_source.columns)-1], axis=1).to_numpy()
y_source = np.array(df_source['class'].astype(int))


x_target = df_target_train.drop(df_target_train.columns[len(df_target_train.columns)-1], axis=1).to_numpy()
x_target_test = df_target_test.drop(df_target_test.columns[len(df_target_test.columns)-1], axis=1).to_numpy()

tfidf = TfidfTransformer()
tfidf.fit(np.concatenate((x_target, x_source), axis=0))
x_source_v = tfidf.transform(x_source).toarray()
x_target_v = tfidf.transform(x_target).toarray()
x_target_test_v = tfidf.transform(x_target_test).toarray()

y_target = np.array(df_target_train['class'].astype(int))
y_target_test = np.array(df_target_test['class'].astype(int))


estimators = [
    # ('dtreegini', DecisionTreeClassifier(max_features="log2", splitter="random", criterion="gini")),
    # ('dtree5', DecisionTreeClassifier(max_depth=5, random_state=1),),
    # ('dtree10', DecisionTreeClassifier(max_depth=10, random_state=1)),
    # ('gnb', GaussianNB()),
    # ('lsvc', NaivelyCalibratedLinearSVC()),
    ('lr', LogisticRegression()),
    # ('svmlight', SVMLight(type="classification", kernel="linear"))
]


tradaboost_n_estimators = [50]
kmm_kernels = ['linear', 'rbf', 'poly']  # adapt
run_tradaboost = False  # both
run_fe = False  # adapt
run_coral = True  # adapt
run_msda = True  # both
run_regular_transfer = True  # adapt
run_kmm = True  # adapt

for ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:

    print(f"Ratio : {ratio}")

    run_experiment_adapt(
        x_source_v, y_source,
        x_target_v, y_target,
        x_target_test_v, y_target_test,
        estimators,
        run_tradaboost,
        run_kmm,
        kmm_kernels,
        run_fe,
        run_coral,
        run_msda,
        run_regular_transfer,
        ratio,
        csv_file,
        tfidf=False
    )


    run_experiment_toolkit(
        x_source_v, y_source,
        x_target_v, y_target,
        x_target_test_v, y_target_test,
        estimators,
        ratio,
        csv_file,
        tfidf=False
    )

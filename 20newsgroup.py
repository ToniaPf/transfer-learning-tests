import csv
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    removed = []
    for r in words:
        if r not in stop_words:
            removed.append(r)
    result = ' '.join(removed)
    return result


csv_file = f'20newsgroup_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'

with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Estimator', 'Ratio', 'Test-Type', 'Accuracy'])


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

y_source = np.array([1 if x >= 2 else 0 for x in source.target])
x_source = np.array(source.data)

df_source = pd.DataFrame({'x_source': x_source, 'y_source': y_source}, columns=['x_source', 'y_source'])
df_source = df_source[df_source.x_source != '']
df_source = df_source[df_source.x_source != '\n']

x_source = np.array(df_source['x_source'])
y_source = np.array(df_source['y_source'])

target = fetch_20newsgroups(
    categories=[
        # 'sci.crypt',
        # 'sci.electronics',
        'sci.med',
        'sci.space',
        # 'talk.politics.guns',
        # 'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc'
    ],
    remove=('headers', 'footers', 'quotes')
)

y_target_init = np.array([1 if x >= 2 else 0 for x in target.target])
x_target_init = np.array(target.data)

df_target = pd.DataFrame({'x_target': x_target_init,
                          'y_target': y_target_init},
                         columns=['x_target', 'y_target'])

df_target = df_target[df_target.x_target != '']
df_target = df_target[df_target.x_target != '\n']

## minority class has 820 instances so selected 800 for both classes
df_target_train = df_target.groupby('y_target').head(600)
df_target_test = df_target.groupby('y_target').tail(200)

x_target = df_target_train['x_target'].to_numpy()
y_target = np.array(df_target_train['y_target'])

x_target_test = df_target_test['x_target'].to_numpy()
y_target_test = np.array(df_target_test['y_target'])

source_size = 2_200


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""
    def fit(self, X, y, sample_weights=None):
        super().fit(X, y, sample_weights)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


estimators = [
    ('dtreegini', DecisionTreeClassifier(max_features="log2", splitter="random", criterion="gini")),
    # ('dtree5', DecisionTreeClassifier(max_depth=5, random_state=1),),
    # ('dtree10', DecisionTreeClassifier(max_depth=10, random_state=1)),
    # ('gnb', GaussianNB()),
    # ('lsvc', NaivelyCalibratedLinearSVC()),
    # ('lr', LogisticRegression()),
    # ('svmlight', SVMLight(type="classification", kernel="linear"))
]

tradaboost_n_estimators = [50]
kmm_kernels = ['linear', 'rbf', 'poly']  # adapt
run_tradaboost = True  # both
run_fe = False  # adapt
run_coral = False  # adapt
run_msda = False  # both
run_regular_transfer = False  # adapt
run_kmm = False  # adapt


for ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:

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

    run_experiment_toolkit(
        x_source, y_source,
        x_target, y_target,
        x_target_test, y_target_test,
        estimators,
        ratio,
        csv_file
    )


import copy
import csv
import numpy as np
from adapt.feature_based._deep import accuracy

from adapt.instance_based import TrAdaBoost, KMM, KLIEP
from adapt.feature_based import FE, mSDA, CORAL, ADDA, DeepCORAL, MCD
from adapt.parameter_based import RegularTransferLC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



def run_experiment_adapt(
    x_source, y_source,
    x_target, y_target,
    x_target_test, y_target_test,
    estimators=[],
    run_tradaboost=True,
    run_kmm=False,
    kmm_kernels=[],
    run_fe=False,
    run_coral=False,
    run_msda=False,
    run_regular_transfer=False,
    ratio=1,
    csv_file='',
    tfidf=True
):

    with open(csv_file, 'a') as f:

        writer = csv.writer(f)

        # kf = KFold(n_splits=3, shuffle=True, random_state=10)
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
                acc = 100 * accuracy_score(y_target_test, preds1)
                print(acc)
                writer.writerow([f'{name}', str(ratio), 'test1', acc])

                ## Test2 --> Train source+target --> Test target ###
                est = copy.copy(base_estimator)
                est.fit(x_source_v, y_source)
                both_x_train = np.concatenate((x_target_train_v, x_source_v), axis=0)
                both_y_train = np.concatenate((y_target_train, y_source), axis=0)
                est.fit(both_x_train, both_y_train)
                preds2 = est.predict(x_target_test_v)
                acc = 100 * accuracy_score(y_target_test, preds2)
                print(acc)
                writer.writerow([f'{name}', str(ratio), 'test2', acc])

                ## Test3 --> Train target --> Test target ###
                est = copy.copy(base_estimator)
                est.fit(x_target_train_v, y_target_train)
                preds3 = est.predict(x_target_test_v)
                acc = 100 * accuracy_score(y_target_test, preds3)
                print(acc)
                writer.writerow([f'{name}', str(ratio), 'test3', acc])


                if run_tradaboost:
                    tr = TrAdaBoost(copy.copy(base_estimator), 50)
                    tr.fit(x_source_v, y_source, x_target_train_v, y_target_train)
                    preds = tr.predict(x_target_test_v)
                    writer.writerow([f'{name}', str(ratio), 'TrAdaBoost', 100 * accuracy_score(y_target_test, preds)])
                    print(f"TrAdaBoost {100 * accuracy_score(y_target_test, preds)}")


                if run_kmm:
                    for kernel in kmm_kernels:
                        kmm = KMM(estimator=base_estimator, kernel=kernel, verbose=0)
                        kmm.fit(x_source_v, y_source, x_target_train_v)
                        preds = kmm.predict(x_target_test_v)
                        acc = 100 * accuracy_score(y_target_test, preds)
                        writer.writerow(
                            [f'{name}', str(ratio), f'KMM - {kernel}', acc])
                        print(f'KMM - {kernel} {acc}')

                if run_fe:
                    fe = FE(base_estimator)
                    fe.fit(x_source_v, y_source, x_target_train_v, y_target_train)
                    preds = fe.predict(x_target_test_v)
                    acc = 100 * accuracy_score(y_target_test, preds)
                    writer.writerow(
                        [f'{name}', str(ratio), f'FE', acc])
                    print(f'FE {acc}')

                if run_coral:
                    try:
                        coral = CORAL(base_estimator)
                        coral.fit(x_source_v, y_source, x_target_train_v)
                        preds = coral.predict(x_target_test_v)
                        acc = 100 * accuracy_score(y_target_test, preds)
                        writer.writerow(
                            [f'{name}', str(ratio), f'Coral', acc])
                        print(f'CORAL {acc}')
                    except BaseException as e:
                        print('error in CORAL')
                        print(e)

                if run_msda:
                    try:
                        m = mSDA(estimator=base_estimator)
                        m.fit(x_source_v, y_source, x_target_train_v)
                        preds = m.predict(x_target_test_v)
                        acc = 100 * accuracy_score(y_target_test, preds)
                        writer.writerow(
                            [f'{name}', str(ratio), f'mSDA', acc])
                        print(f'MSDA {acc}')
                    except BaseException as e:
                        print('error in MSDA')
                        print(e)

                if run_regular_transfer:
                    if name == "lr":
                        try:

                            print(f'### regular transfer {name} ###')

                            e = copy.copy(est)
                            rt = RegularTransferLC(e)

                            rt.fit(x_target_train_v, y_target_train)
                            preds = rt.predict(x_target_test_v)
                            print(100 * accuracy_score(y_target_test, preds))
                            writer.writerow(
                                [f'{name}', str(ratio), f'regular_transfer', 100*accuracy_score(y_target_test, preds)])
                        except BaseException as e:
                            print('error in regular transfer')
                            print(e)

                # if run_kliep := True:
                #     print(f'### KLIEP {name} ###')
                #
                #     kliep = KLIEP(base_estimator(**kwargs))
                #     kliep.fit(x_source_v, y_source, x_target_train_v)
                #     preds = kliep.predict(x_target_test_v)
                #     print(100 * accuracy(y_target_test, preds))
                #     writer.writerow(
                #         [f'{name}', str(ratio), f'kliep', 100 * accuracy_score(y_target_test, preds)])

                # if run_deep_coral := False:
                #     print(f'### deepCoral ###')
                #     deepcoral = DeepCORAL(lambda_=0.)
                #     deepcoral.fit(x_source_v, y_source, x_target_train_v, y_target_train, epochs=200)
                #     preds = deepcoral.predict(x_target_test_v)
                #     print(100 * accuracy(y_target_test, preds))
                #     writer.writerow(
                #         [f'{name}', str(ratio), f'deepcoral', 100 * accuracy_score(y_target_test, preds)])
                #
                # if run_adda := False:
                #     print(f'### ADDA ###')
                #     adda = ADDA()
                #     adda.fit(x_source_v, y_source, x_target_train_v, y_target_train, epochs=20)
                #     preds = adda.predict(x_target_test_v)
                #     print(100 * accuracy(y_target_test, preds))
                #     writer.writerow(
                #         [f'{name}', str(ratio), f'adda', 100 * accuracy_score(y_target_test, preds)])
                #

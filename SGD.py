from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from AccuracyHelper import *
import numpy as np
from warnings import filterwarnings
import time
import joblib

# Stochastic Gradient Descent with log loss
# Essentially running logistic regression
def SGD(X, y, X_test, y_test, refit=False):
    start = time.time()

    if refit:
        filterwarnings('ignore')
        lin_clf = SGDClassifier(loss='log', max_iter=4000, n_jobs=-1)

        # Fine-tune
        print("Running GridSearch on SGD Model...")
        param_grid = {'alpha':[0.01, 0.001, 0.0001],
             'epsilon':[0.1, 0.01, 0.001, 0.0001],
             'l1_ratio':[.1, .25, .5],
             'tol':[.01, .001, .0001, .00001]}
        grid_clf = GridSearchCV(lin_clf, param_grid, cv=3, n_jobs=-1)

        print("Fitting the data...")
        grid_clf.fit(X, y)

        print("Saving model... ")
        joblib.dump(grid_clf, "models/sgd.sav")
    else:
        print("Loading model...")
        grid_clf = joblib.load("models/sgd.sav")

    print(f"Getting predictions...")
    y_pred_lin = grid_clf.predict(X_test)

    # a = 0
    # for guess_lin, actual in zip(y_pred_lin, y_test):
    #     if guess_lin == actual:
    #         a += 1
    accuracy = CalculateAccuracy(X_test, y_test, y_pred_lin)
    a = accuracy[-1,1]
    DisplayAccuracy(accuracy)

    end = time.time()
    print(f"SGD Loss: {accuracy[-1,0]}\t Time:{end-start:.4f}s\n")

    return accuracy

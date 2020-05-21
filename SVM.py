from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from AccuracyHelper import *
import numpy as np
from warnings import filterwarnings
import time
import joblib

# Support vector machine
def SVM(X, y, X_test, y_test, refit=False):
    start = time.time()

    if refit:
        filterwarnings('ignore')
        lin_clf = svm.SVC(max_iter=2000)

        # Fine-tune
        print("Running GridSearch on SVM Model...")
        param_grid = {'C':[1, 10, 100],
             'gamma':['scale', 'auto'],
             'tol':[.01, .001, .0001, .00001]}
        grid_clf = GridSearchCV(lin_clf, param_grid, cv=3, n_jobs=-1)

        print("Fitting the data...")
        grid_clf.fit(X, y)

        print("Saving model...")
        joblib.dump(grid_clf, "models/svm.sav")
    else:
        print("Loading model...")
        grid_clf = joblib.load("models/svm.sav")

    print(f"Getting predictions...")
    y_pred_lin = grid_clf.predict(X_test)

    accuracy = CalculateAccuracy(X_test, y_test, y_pred_lin)
    a = accuracy[-1,1]
    DisplayAccuracy(accuracy)

    end = time.time()
    print(f"SVM Loss: {accuracy[-1,0]}\t Time:{end-start:.2f}s\n")

    return accuracy

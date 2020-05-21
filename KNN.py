from sklearn.neighbors import KNeighborsClassifier
from AccuracyHelper import *
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt

def KNN(X, y, X_test, y_test, refit=False):
    ks = range(2,7)
    a = [0 for x in range(5)]

    # for k=2...10, run KNN classifier
    accuracies = []; end = []; start = [];
    for k in ks:
        start.append(time.time())
        print(f"Running KNN with k={k}...")

        if refit:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
            print("Fitting the data...")
            knn.fit(X, y)

            print("Saving model... ")
            joblib.dump(knn, f"models/knn-{k}.sav")
        else:
            print("Loading model...")
            knn = joblib.load(f"models/knn-{k}.sav")

        print("Getting predictions...")
        y_pred = knn.predict(X_test)

        print("Calculating loss...")
        accuracy = CalculateAccuracy(X_test, y_test, y_pred)
        a[k - 2] = accuracy[-1,1]
        DisplayAccuracy(accuracy)
        accuracies.append(accuracy)
        end.append(time.time())
        print(f"Loss: {accuracy[-1,0]}\t Time: {end[-1]-start[-1]:.2f}s\n")

    # errors for k=2...10
    a = np.array(a)
    best = np.argmax(a)

    fig, ax = plt.subplots()
    PlotKNN(ks, accuracies, np.subtract(end, start), ax)
    fig.tight_layout()
    plt.show()

    # take the best
    return accuracies[best], best+2, start[best], end[best]


def PlotKNN(ks, accuracies, efficiencies, ax):
    acc = [x[-1,1] for x in accuracies]
    x = np.arange(len(ks))

    ax2 = ax.twinx()
    ax2.plot(x, efficiencies, linestyle='dotted', label="Efficiencies", color = 'g')
    ax2.set_ylabel("Efficiency (s)", color = 'g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax.plot(x, acc, linestyle='dashed', label="Accuracies", color = 'b')
    ax.set_ylabel("Accuracy (%)", color = 'b')
    ax.tick_params(axis='y', labelcolor='b')

    ax.set_title("Accuracy & Efficiency with different K's")
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_xlabel("k")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

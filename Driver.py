import time
import sys
from KNN import KNN
from SGD import SGD
from SVM import SVM
import NN
from DataLoader import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from AccuracyHelper import *

def main():
    # Handle refit argument
    refit = False
    if len(sys.argv) > 2:
        print("Too many arguments... check out the readme.")
        exit()
    if len(sys.argv) == 2:
        if string(sys.argv)[1] != "--refit":
            print("Unrecognized argument passed... check out the readme.")
            exit()
        else:
            print("Refit is turned on!")
            refit = True

    # load in training and test data
    print("Loading Data...")
    train = DataLoader("training")
    validation = DataLoader("validation")
    test = DataLoader("test")
    test = np.vstack((test,validation))

    # split the data
    X = train[:, :-1]
    y = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    print("Data Loaded.\n")

    # PCA and Scaling
    print(f"Reducing dimensionality of data with PCA...")
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=100, random_state=0))
    pca.fit(X, y)
    pca_model = pca.named_steps['pca']
    print("{0:.2f}% of variance explained\n"
            .format(pca_model.explained_variance_ratio_.cumsum()[-1]*100))
    X = pca.transform(X)
    X_test = pca.transform(X_test)

    # Experimentation
    accuracies = []; end = []; start = [];

    print("-= Begin KNN =-")
    kstart = time.time()
    accuracy, k, st, nd = KNN(X, y, X_test, y_test, refit)
    kend = time.time()
    start.append(st)
    end.append(nd)
    accuracies.append(accuracy)
    print(f"KNN acc:{100 * accuracy[-1,1]:.2f}%\t total time:{(kend-kstart):.2f}s\t best k:{k}\n")

    print("-= Begin SVM =-")
    start.append(time.time())
    accuracy = SVM(X, y, X_test, y_test, refit)
    end.append(time.time())
    accuracies.append(accuracy)
    print(f"SVM acc:{100 * accuracy[-1,1]:.2f}%\t total time:{(end[-1]-start[-1]):.2f}s\n")

    print("-= Begin SGD =-")
    start.append(time.time())
    accuracy = SGD(X, y, X_test, y_test, refit)
    end.append(time.time())
    accuracies.append(accuracy)
    print(f"SGD acc:{100 * accuracy[-1,1]:.2f}%\t total time:{(end[-1]-start[-1]):.4f}s\n")

    print("-= Begin NN =-")
    start.append(time.time())
    accuracy = NN.test_model("models/nn-vgg4.h5", "vgg4")
    end.append(time.time())
    accuracies.append(accuracy)
    print(f"SVM acc:{100 * accuracy[-1,1]:.2f}%\t total time:{(end[-1]-start[-1]):.2f}s\n")

    # Plot Sumary Visualizations
    np.array(accuracies)
    fig, ax = plt.subplots()
    PlotClassAccuracy(accuracies, ax)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    PlotAlgAccEff(accuracies, np.subtract(end, start), ax)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

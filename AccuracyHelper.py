import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

def CalculateAccuracy(X, y_act, y_pred):
    class_counts = np.bincount(y_act)
    n_classes = len(class_counts)

    # for each class [loss, accuracy]
    # last entry is for whole dataset
    accuracy = np.zeros((n_classes + 1, 2))

    total = X.shape[0]
    total_correct = np.count_nonzero(y_act == y_pred)
    for i in range(n_classes):
        class_correct = 0
        for guess, actual in zip(y_pred, y_act):
            if actual == i and guess == actual:
                class_correct += 1
        accuracy[i,0] = class_counts[i] - class_correct
        accuracy[i,1] = class_correct / class_counts[i]

    accuracy[-1,0] = total - total_correct
    accuracy[-1,1] = total_correct / total

    return accuracy

def DisplayAccuracy(accuracy):
    classes = os.listdir("data/images/test")
    classes.sort()
    for i in range(len(accuracy) - 1):
        acc = accuracy[i]
        print(f"{classes[i]:18} Loss: {acc[0]:<4.0f} Accuracy: {acc[1]*100:.2f}%")
    print()

def PlotClassAccuracy(accuracies, ax):
    knn = accuracies[0]
    svm = accuracies[1]
    sgd = accuracies[2]
    nn = accuracies[3]

    classes = os.listdir("data/images/test")
    classes.sort()
    x = np.arange(len(classes))
    width = 0.20
    ax.barh(x + 1.5*width, knn[:-1,-1], width, label="KNN", color="#00E6CF")
    ax.barh(x + width/2, svm[:-1,-1], width, label="SVM", color="#00BFAF")
    ax.barh(x - width/2, sgd[:-1,-1], width, label="SGD", color="#008075")
    ax.barh(x - 1.5*width, nn[:-1,-1], width, label="NN", color="#00403A")

    ax.set_xlabel("Accuracy")
    ax.set_title("Per Class Accuracy on Different Classifiers")
    ax.set_yticks(x)
    ax.set_yticklabels(classes)
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.legend()

def PlotAlgAccEff(accuracies, efficiencies, ax):
    acc = [x[-1,1] for x in accuracies]
    x = np.arange(4)
    algs = ["KNN", "SVM", "SGD", "NN"]

    ax2 = ax.twinx()
    b1 = ax2.bar(x+0.4, efficiencies, .4, label="Efficiencies", color = 'g')
    ax2.set_ylabel("Efficiency (s)", color = 'g')
    ax2.tick_params(axis='y', labelcolor='g')

    b2 = ax.bar(x, acc, .4, label="Accuracies", color = 'b')
    ax.set_ylabel("Accuracy (%)", color = 'b')
    ax.tick_params(axis='y', labelcolor='b')

    ax.set_title("Accuracy & Efficiency on Different Classifiers")
    ax.set_xticks(x)
    ax.set_xticklabels(algs)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    def labelbar(bars,a):
        for b in bars:
            height = b.get_height()
            a.annotate('{:.2f}'.format(height), xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    labelbar(b1, ax2)
    labelbar(b2, ax)

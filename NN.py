import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # turn off annoying tensorflow logs
import tensorflow.keras
import numpy as np
from DataLoader import DataLoaderNN
from AccuracyHelper import *
import time

def test_model(model_path, model_name, verbose=False):
    classifier = tensorflow.keras.models.load_model(model_path)
    input_shape = classifier.input_shape[1:4]  # For some reason, input-shape has the form (None, 1, 2, 3)

    if verbose:
        print("Model: ", model_name)
        print("Weights loaded from : ", model_path)

        print("Loading classifier...")
        classifier.summary()

        print(" Input shape: {0}, Output: {1} classes".format(input_shape, classifier.output_shape[1]))

    X, y = DataLoaderNN("test")

    start = time.time()
    results = classifier.predict(X)
    y_pred = np.argmax(results, axis=1)

    accuracy = CalculateAccuracy(X, y, y_pred)
    DisplayAccuracy(accuracy)
    end = time.time()

    print(f"NN Loss: {accuracy[-1,0]}\t Time:{end-start:.4f}s\n")
    return accuracy

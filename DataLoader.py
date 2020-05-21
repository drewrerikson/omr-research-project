import glob
import os

from PIL import Image
import numpy as np

def DataLoader(dataset):
    if dataset != "training" and dataset != "validation" and dataset != "test":
        raise ValueError("expected dataset to be {training | validation | test}, got "+ dataset)

    datapath = "data/images/" + dataset
    classes = os.listdir(datapath)
    classes.sort()
    images = []
    for f in glob.iglob(datapath + "/*/*"):
        class_name = f.split("/")[-2]
        y = classes.index(class_name)
        img = np.array(Image.open(f).convert("L")).flatten() // 255
        images.append(np.concatenate((img,[y])))
    return np.array(images)

def DataLoaderNN(dataset):
    if dataset != "training" and dataset != "validation" and dataset != "test":
        raise ValueError("expected dataset to be {training | validation | test}, got "+ dataset)

    datapath = "data/images/" + dataset
    classes = os.listdir(datapath)
    classes.sort()
    images = []
    y = []
    for f in glob.iglob(datapath + "/*/*"):
        class_name = f.split("/")[-2]
        y.append(classes.index(class_name))
        images.append(np.array(Image.open(f).convert("RGB")))
    return np.array(images), np.array(y)

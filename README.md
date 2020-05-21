# CSCI 4511 OMR Project

### Data
Utilizing the HOMUS dataset (https://grfia.dlsi.ua.es/homus/) with 15200 symbols spread across 32 classes. Each image has been shrunk to 112x112 pixels. The data is split into training (80%), validation (10%), and test (10%) subsets.

Unpack `homus_data.zip` into `data/` and create directory `models/`. The file system should be such that
```
csci-4511w-omr-proj
├── data
│   └── images
│       ├── 12-8-Time
|       :
```

### Installing Dependencies:

```bash
  conda|pip3 install numpy
  conda|pip3 install sklearn
  conda|pip3 install pillow
  conda|pip3 install joblib
```

### Running It:

```bash
  python3 Driver.py [--refit]
```

Note that the `refit` argument will significantly heighten run-time when provided to the application. If a model is fitted, it will be saved to `models/`, and the GridSearch portion can be skipped by omitting this parameter. **If you are running this program for the first time, you need to generate the models and need to have this turned on!**

The neural network model must be trained from https://github.com/apacha/MusicSymbolClassifier. We used the following command to generate our model.
```bash
python3 ModelTrainer/TrainModel.py --datasets homus --width 112 --height 112 --minibatch_size 32 --model_name vgg4
```

### Todo:
* [x] Split by Class
* [x] Pickle models
* [x] Fine-Tuning/Grid-Search
* [x] Efficiency & Accuracy Graph
* [x] Split by Class Side Bar Graph
* [x] KNN Graph
* [x] *Record results*
* [x] *Write the paper*


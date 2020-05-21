# CSCI 4511 OMR Final Project

Drew and Carter try to be good at OMR. _We dem boys._

### Data
Utilizing the HOMUS dataset [https://grfia.dlsi.ua.es/homus/] with 15200 symbols spread across 32 classes. Each image has been shrunk to 112x112 pixels. The data is split into training (80%), validation (10%), and test (10%) subsets.

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

### Todo:
* [x] Split by Class
* [x] Pickle models
* [x] Fine-Tuning/Grid-Search
* [x] Efficiency & Accuracy Graph
* [x] Split by Class Side Bar Graph
* [x] KNN Graph
* [ ] *Record results*
* [ ] *Write the damn paper*

### Timeline:
* [x] Week 1:
  - Project Proposal, Literature Review, Collection of Sources, Experimental Definition, Begin Implementation
* [x] Week 2:
  - Completely Implement Framework for Experiments (Write scripts for running and collecting data for KNN, SVM, and Deep Learning approaches)
* [ ] Week 3:
  - Run Experiments, Collect Data, Begin Analysis and Report
* [ ] Week 4:
  - Continue Report, Analyze Results, Draw Conclusions, Draw up Figures
* [ ] Week 5: *(If needed)*
  - Finish Report, Write Abstract, Final Revising, Turn-in

### Credits
- Carter Mintey
- Drew Erikson
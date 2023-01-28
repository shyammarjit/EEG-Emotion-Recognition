## Structure of Code
```text
├── main.py                                 <- main code (only this script needs to run)
│   ├── data_preprocessing.py               <- Data Preprocessing [Band Pass Filtering -> ICA -> CAR]
│   ├── MultiDomain_Feature_Extraction.py   <- Multi-Domain features extraction
│       ├── utils.py                        <- Functions for feature extraction
│   ├── feature_selection.py                <- Genetic Algorithm for feature selection
│   ├── GAMLP.py                            <- Genetic Algorithm optimized Multi-Layer Perceptron
```


**✨ Note for readers:**<br/>
The value of the hyperparameters are kept to just check the running status of the code. You may need to tune the hyperparameters.<br/>
Thank you for reading my paper.


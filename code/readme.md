## Structure of Code
```text
├── main.py                                 <- main code (only this script needs to run)
│   ├── data_preprocessing.py               <- Data Preprocessing [Band Pass Filtering -> ICA -> CAR]
│   ├── MultiDomain_Feature_Extraction.py   <- Multi-Domain features extraction
│       ├── utils.py                        <- Functions for feature extraction
│   ├── feature_selection.py                <- Genetic Algorithm for feature selection
│   ├── GAMLP.py                            <- Genetic Algorithm optimized Multi-Layer Perceptron
```

**✨ Description of few imp variables:**
```
dataset_path  <- Path for the DEAP dataset
subject_no    <- Subject number in string format for e.g. "s02", "s12"

# GA variables for Feature selection (FS)
GAFS_numPop       <- Size of the population in GAFS
GAFS_numGen       <- No of generations in GAFS
GAFS_cross_prob   <- Crossover probability in GAFS
GAFS_mut_probb    <- Mutation probability in GAFS
alpha             <- Weight for classification accuracy in fitness function
	
# GA variables for GA-MLP
GAMLP_numPop      <- Size of the population in GA-MLP
GAMLP_numGen      <- No of generations in GA-MLP
GAMLP_prob_cross  <- Crossover probability in GA-MLP
GAMLP_prob_mut    <- Mutation probability in GA-MLP
```

**✨ Note for readers:**<br/>
The value of the hyperparameters are kept to just check the running status of the code. You may need to tune the hyperparameters.<br/>
Thank you for reading my paper.


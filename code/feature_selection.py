import random
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import accuracy_score
from deap import creator, base, tools, algorithms
from sklearn.linear_model import LogisticRegression


def getFitness(individual, data, alpha = 0.90):
    selected_features = []
    total_features = int(data['trainX'].shape[1]) # total no of features
    features = list(data['trainX'].columns) # get all the feature names
    if(len(set(individual)) == 1 and list(set(individual))[0] == 0):
        # If all gene values are 0 then return 0
        return (0,)
    for i in range(0, len(individual)):
        if(individual[i]==1):
            selected_features.append(features[i])
    no_sel_feat = len(selected_features)
    _classifier = LogisticRegression() # classifier
    new_trainX = data['trainX'][selected_features].copy()
    new_testX = data['testX'][selected_features].copy()
    _classifier.fit(new_trainX, data['trainY'])
    predictions = _classifier.predict(new_testX)
    accuracy = accuracy_score(y_true = data['testY'], y_pred = predictions)
    my_fitness = alpha*accuracy + (1-alpha)*((total_features - no_sel_feat)/total_features)
    return (my_fitness,)

def GAFS(data, numPop = 100, numGen = 50, cross_prob = 0.65, mut_prob = 0.15, alpha = 0.90):
    '''
    Genetic Algorithm based feature selction.
    data: a tuple
    numPop, numGen, and alpha are the hyperparameters.
    '''
    # setting up the configurations
    creator.create('FitnessMax', base.Fitness, weights = (1.0,))
    creator.create('Individual', list, fitness = creator.FitnessMax)
    toolbox = base.Toolbox() # Create Toolbox
    toolbox.register('attr_bool', random.randint, 0, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool,\
                     int(data['trainX'].shape[1]))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', getFitness, data = data)
    toolbox.register('mate', tools.cxOnePoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb = 0.1)
    toolbox.register('select', tools.selTournament, tournsize = 7)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Initialize population and hof
    population = toolbox.population(numPop)
    hof = tools.HallOfFame(numPop * numGen)
    
    # Launch genetic algorithm, change the crossover and mutation probability
    population, log_file = algorithms.eaSimple(population, toolbox, cxpb = cross_prob, mutpb = mut_prob,\
        ngen=numGen, stats=stats, halloffame=hof, verbose=False)
    
    return population[0] # return the most optimal feature subset

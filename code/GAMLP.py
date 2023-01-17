import random
import numpy as np
from tqdm import trange
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def crossover(parent_1, parent_2, prob_cross):
    # perform crossover
    # parents: [a1, b1] & [a2, b2]
    # offspring: [a1, b2] & [a2, b1]
    if(prob_cross>=random.uniform(0, 1)):
        return [parent_1[0], parent_2[1], parent_1[2], parent_2[3]]
    else:
        return [parent_1[0], parent_1[1], parent_2[2], parent_2[3]]

def mutation(offsprings, prob_mut):
    # perform mutation
    for ind in range(0, len(offsprings)):
        if np.random.rand() <= prob_mut:
            act_or_solver = random.randint(2, 3)
            random_noise = random.randint(1, 4)
            offsprings[ind][act_or_solver] = int(offsprings[ind][act_or_solver]) + random_noise
    return offsprings

def compute_fitness(population, data):
    fitness = []
    for config in population:
        clf = MLPClassifier(learning_rate_init=0.09, activation=config[0], solver = config[1], alpha=1e-5,\
                            hidden_layer_sizes=(int(config[2]), int(config[3])),\
                            max_iter=1000, n_iter_no_change=80)
        clf.fit(data['trainX'], data['trainY'])
        fitness.append([accuracy_score(clf.predict(data['testX']), data['testY']), clf, list(config)])
    return np.array(fitness)

def initialize_population(pop_size):
    # Initialize the population
    activations = ['identity','logistic', 'tanh', 'relu']
    optimizers = ['lbfgs', 'sgd', 'adam']
    population = []
    for i in range(0, pop_size):
        rand_act = random.choice(activations)
        rand_opt = random.choice(optimizers)
        neurons_1st = random.randint(2, 100) # no of neurons in 1st layers
        neurons_2nd = random.randint(2, 50) # no of neurons in 2nd layers
        population.append(np.array([rand_act, rand_opt, neurons_1st, neurons_2nd]))
    return np.array(population)

def GA_MLP(data, generations = 10, pop_size = 20, prob_cross = 0.95, prob_mut=0.15):
    population = initialize_population(pop_size) # Initialize new population
    pop_fitness = compute_fitness(population, data) # find the fitness of the population
    # sort the indivisuals based on their fitness function
    pop_sorted = np.array(list(reversed(sorted(pop_fitness, key=lambda ind: ind[0]))))
    for gen in range(0, generations):
        # parent selection
        parent_1 = pop_sorted[:,2][:pop_size//2] # take 1st half as parent-1
        parent_2 = pop_sorted[:,2][pop_size//2:] # take 2nd half as parent-2
        # crossover
        offsprings_1 = [crossover(parent_1[i], parent_2[i], prob_cross) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        offsprings_2 = [crossover(parent_2[i], parent_1[i], prob_cross) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        # mutation
        offsprings_1 = mutation(offsprings_1, prob_mut)
        offsprings_2 = mutation(offsprings_2, prob_mut)
        
        # calculates the fitness of the children to choose who will pass on to the next generation
        offsprings_1 = compute_fitness(offsprings_1, data)
        offsprings_2 = compute_fitness(offsprings_2, data)
        # concatenate all parents and new generated offsprings
        new_population = np.concatenate((pop_sorted, offsprings_1, offsprings_2))
        # sorted the merged population
        new_population = np.array(list(reversed(sorted(new_population, key=lambda x: x[0]))))
        # select individuals of the next generation with the same population size
        pop_sorted = new_population[0:pop_size, :]
    # return the predicted labels, trained with best classifier among all the generations
    return pop_sorted[0][1].predict(data['testX'])
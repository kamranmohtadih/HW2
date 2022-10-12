import mlrose_hiive as ml
import numpy as np
import time
import matplotlib.pyplot as plt


# ***************************************************
# Problems to solve *********************************
# 8 Queens, 4 peaks and flip flop problems **********
# ***************************************************
problems =[]
def queens_max(state):
    fitness_cnt = 0
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            if (state[j] != state[i]) \
                    and (state[j] != state[i] + (j - i)) \
                    and (state[j] != state[i] - (j - i)):
                fitness_cnt += 1
    return fitness_cnt


fitness_1 = ml.CustomFitness(queens_max)
problem_1 = ml.DiscreteOpt(length = 8, fitness_fn = fitness_1, maximize = True, max_val = 8)
fitness_2 = ml.FourPeaks(t_pct=0.15)
problem_2 = ml.DiscreteOpt(length=8, fitness_fn=fitness_2, maximize = True)
fitness_3 = ml.FlipFlop()
problem_3 = ml.DiscreteOpt(length=8, fitness_fn=fitness_3, maximize = True)

problems.append(problem_1)

problems.append(problem_2)
problems.append(problem_3)

# ******************************************************************************
# Algorithms to apply **********************************************************
# randomized hill climbing, simulated annealing, genetic algorithm and Mimic ***
# ******************************************************************************

def process(p):
    best_fitnesss_1 = []
    durations_1 = []

    schedule = ml.ExpDecay()
    for iter in range(1000):
        start = time.time()
        best_state, best_fitness , curve= ml.simulated_annealing(problem= p, max_attempts=100,max_iters=iter,schedule=schedule,random_state=666)
        end = time.time()
        duration = end - start
        best_fitnesss_1.append(best_fitness)
        durations_1.append(duration)
    plt.plot(range(1000), best_fitnesss_1, label="Simulated annealing")
    plt.xlabel('Iterations')
    plt.ylabel('Best fitness')
    plt.legend()
    plt.show()
    plt.savefig('images/Simulated_annealing-best_fitness.png')
    plt.clf()
    plt.close('images/Simulated_annealing-best_fitness.png')


    pop_size = [100,200,300,400,500]
    mut_prob = [0.1,0.2,0.3]
    for ps in pop_size:
        for mp in mut_prob:
            best_fitnesss_2 = []
            durations_2 = []
            for iter in range(1000):
                start = time.time()
                best_state, best_fitness , curve= ml.genetic_alg(p, max_attempts=100,
                                                       max_iters=iter,
                                                       pop_size=ps,
                                                       mutation_prob=mp,
                                                       random_state=666)
                end = time.time()
                duration = end - start
                best_fitnesss_2.append(best_fitness)
                durations_2.append(duration)
            plt.plot(range(1000), best_fitnesss_2, label="Genetic algorithm mut_prop="+mp+" pop_size="+ps)
            plt.xlabel('Iterations')
            plt.ylabel('Best fitness')
    plt.legend()
    plt.savefig('images/Gen_alg-best_fitness.png')
    plt.clf()
    plt.close('images/Gen_alg-best_fitness.png')


    best_fitnesss_3 = []
    durations_3 = []
    restarts = [10, 20, 30, 50, 100]

    for i in restarts:
        start = time.time()
        state, fitness , curve= ml.random_hill_climb(problem=p, max_attempts=100, max_iters=1000, restarts=i,
                                                     curve=True, random_state=666)
        end = time.time()
        duration = end - start
        best_fitnesss_3.append(best_fitness)
        durations_3.append(duration)

    best_fitnesss_4 = []
    durations_4 = []
    pop_size = [100,200,300,400,500]

    for ps in pop_size:
        start = time.time()
        state, fitness , curve= ml.mimic(problem=p, pop_size=ps, max_attempts=100, max_iters=1000, curve=True, random_state=666)
        end = time.time()
        duration = end - start
        best_fitnesss_4.append(best_fitness)
        durations_4.append(duration)

    return best_fitnesss_1,best_fitnesss_2,best_fitnesss_3,best_fitnesss_4,durations_1,durations_2,durations_3,durations_4


if __name__ == '__main__':
    for p in problems:
        process(p)


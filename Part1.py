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
def process(p,counter):
    myIter = 1000
    myMaxAttempt = 10
    pop_size = 400
    best_fitnesss_1 = []
    durations_1 = []

    schedule = ml.ExpDecay()
    start = time.time()
    for iter in range(myMaxAttempt):
        best_state, best_fitness , curve= ml.simulated_annealing(problem= p, max_attempts=iter,max_iters=myIter,schedule=schedule,random_state=666)
        end = time.time()
        duration = end - start
        best_fitnesss_1.append(best_fitness)
        durations_1.append(duration)
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(range(myMaxAttempt), best_fitnesss_1, 'b--', label='Simulated_annealing')
    ax2.plot(durations_1, best_fitnesss_1, 'b--',label='Simulated_annealing')
    ax1.set_xlabel('Max Attempts')
    ax1.set_ylabel('Fitness function')
    ax1.set_title('Best fitness')
    ax1.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fitness function')
    ax2.set_title('Time taken')
    ax2.legend()

    print('simulated annealing finished...')

    mut_prob = [0.2]
    for mp in mut_prob:
        best_fitnesss_2 = []
        durations_2 = []
        start = time.time()
        for iter in range(myMaxAttempt):
            best_state, best_fitness , curve= ml.genetic_alg(p, max_attempts=iter, pop_size=pop_size,
                                                   max_iters=myIter,
                                                   mutation_prob=mp,
                                                   random_state=666)
            end = time.time()
            duration = end - start
            best_fitnesss_2.append(best_fitness)
            durations_2.append(duration)

        ax1.plot(range(myMaxAttempt), best_fitnesss_2, 'r--',label='Genetic')
        ax2.plot(durations_2, best_fitnesss_2, 'r--',label='Genetic')
        ax1.legend()
        ax2.legend()

    print('Genetic algorithm finished...')


    restarts = [10]

    for i in restarts:
        best_fitnesss_3 = []
        durations_3 = []
        start = time.time()
        for iter in range(myMaxAttempt):
            state, best_fitness , curve= ml.random_hill_climb(problem=p, max_attempts=iter, max_iters=myIter, restarts=i,
                                                         curve=True, random_state=666)
            end = time.time()
            duration = end - start
            best_fitnesss_3.append(best_fitness)
            durations_3.append(duration)

        ax1.plot(range(myMaxAttempt), best_fitnesss_3, 'y--',label='Random hill climb')
        ax2.plot(durations_3, best_fitnesss_3, 'y--',label='Random hill climb')
        ax1.legend()
        ax2.legend()

    print('random hill climbing finished...')
    best_fitnesss_4 = []
    durations_4 = []
    start = time.time()
    for iter in range(myMaxAttempt):
        state, best_fitness , curve= ml.mimic(problem=p, pop_size=pop_size, max_attempts=iter, max_iters=myIter, curve=True, random_state=666)
        end = time.time()
        duration = end - start
        best_fitnesss_4.append(best_fitness)
        durations_4.append(duration)

    ax1.plot(range(myMaxAttempt), best_fitnesss_4, 'k--',label='Mimic')
    ax2.plot(durations_4, best_fitnesss_4, 'k--',label='Mimic')
    ax1.legend()
    ax2.legend()

    plt.savefig('HW2/images/'+str(counter) +'.png')
    plt.clf()
    plt.close('HW2/images/'+str(counter) +'.png')
    print('Mimic finished...')

if __name__ == '__main__':
    process(problems[0],"8 queens")
    process(problems[1], "4 Peaks")
    process(problems[2], "FlipFlop")


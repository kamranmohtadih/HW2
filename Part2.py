import mlrose_hiive as ml
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Read the data and prpare the plot************************************************

df1 = pd.read_csv('HW2/data1.csv', index_col=0, header=0)
df1.replace([np.inf, -np.inf], np.nan, inplace=True)
df1.dropna(inplace=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
names = df1.columns.values
indexes = df1.index.values
x = df1.iloc[:,:-1]
y = df1.iloc[:,-1]
x = preprocessing.OneHotEncoder().fit_transform(x).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
cv = ShuffleSplit(n_splits=3
                  , test_size=0.2, random_state=0)

myIter = 1000
max_attempts = 200
restart = 3
rate = 0.2
pop_size = 50

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
# ************************************************************
# Neural network with randomized hill climbing ***************
# ************************************************************
NN = ml.NeuralNetwork(hidden_nodes=[3,5], activation='sigmoid', algorithm='random_hill_climb', max_iters=myIter,  is_classifier=True,
                        learning_rate=rate, restarts=restart, max_attempts=max_attempts, random_state=666, curve=True, bias=True, pop_size=pop_size)
NN.fit(x_train,y_train)
train_sizes, train_scores, test_scores, fit_times, fit_scores = learning_curve(NN, x, y, train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy',  random_state=666, return_times=True)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
fit_mean = np.mean(fit_times, axis=1)


ax1.plot( NN.fitness_curve[1], 'b--', label='RHC - fitness eval function')
ax2.plot( train_mean, 'b--', label='RHC - training learning curve')
ax2.plot( test_mean, 'b', label='RHC - testing learning curve')
ax3.plot( fit_mean, 'b--', label='RHC - time')
ax1.set_xlabel('iteration')
ax2.set_xlabel('iteration')
ax3.set_xlabel('iteration')
ax1.set_ylabel('Fitness')
ax2.set_ylabel('Accuracy')
ax3.set_ylabel('Time')
ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')

# ************************************************************
# Neural network with simulated annealing ***************
# ************************************************************
NN = ml.NeuralNetwork(hidden_nodes=[3,5], activation='sigmoid', algorithm='simulated_annealing', max_iters=myIter,  is_classifier=True,
                        learning_rate=rate, restarts=restart, max_attempts=max_attempts, random_state=666, curve=True, bias=True, pop_size=pop_size)
NN.fit(x_train,y_train)
train_sizes, train_scores, test_scores, fit_times, fit_scores = learning_curve(NN, x, y, train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy',  random_state=666, return_times=True)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
fit_mean = np.mean(fit_times, axis=1)


ax1.plot( NN.fitness_curve[1], 'r--', label='Simulated Annealing - fitness eval function')
ax2.plot( train_mean, 'r--', label='Simulated Annealing - training learning curve')
ax2.plot( test_mean, 'r', label='Simulated Annealing - testing learning curve')
ax3.plot( fit_mean, 'r--', label='Simulated Annealing - time')
ax1.set_xlabel('iteration')
ax2.set_xlabel('iteration')
ax3.set_xlabel('iteration')
ax1.set_ylabel('Fitness')
ax2.set_ylabel('Accuracy')
ax3.set_ylabel('Time')
ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
# ************************************************************
# Neural network with Genetic algorithm ***************
# ************************************************************
NN = ml.NeuralNetwork(hidden_nodes=[3,5], activation='sigmoid', algorithm='genetic_alg', max_iters=myIter,  is_classifier=True,
                        learning_rate=rate, restarts=restart, max_attempts=max_attempts, random_state=666, curve=True, bias=True, pop_size=pop_size)
NN.fit(x_train,y_train)
train_sizes, train_scores, test_scores, fit_times, fit_scores = learning_curve(NN, x, y, train_sizes=np.linspace(0.2, 1.0, 5), scoring='accuracy',  random_state=666, return_times=True)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
fit_mean = np.mean(fit_times, axis=1)


ax1.plot( NN.fitness_curve[1], 'y--', label='Genetic alg - fitness eval function')
ax2.plot( train_mean, 'y--', label='Genetic alg - training learning curve')
ax2.plot( test_mean, 'y', label='Genetic alg - testing learning curve')
ax3.plot( fit_mean, 'y--', label='Genetic alg - time')
ax1.set_xlabel('iteration')
ax2.set_xlabel('iteration')
ax3.set_xlabel('iteration')
ax1.set_ylabel('Fitness')
ax2.set_ylabel('Accuracy')
ax3.set_ylabel('Time')
ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')

# *************************************************************************
# Print plot **************************************************************
# *************************************************************************
plt.tight_layout()
plt.savefig('HW2/images/NN.png')
plt.clf()
plt.close('HW2/images/NN.png')

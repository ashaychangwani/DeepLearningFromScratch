# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:55:50 2020

@author: Ashay
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv', header = None)

df.columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Class"]

dtypes = list(zip(df.dtypes.index, map(str, df.dtypes)))
for column, dtype in dtypes:            #this is the normalization of data. It helps gradient descent converge much faster and more accurately.
    if dtype == 'float64':
        df[column] -= df[column].mean()
        df[column] /= df[column].std()




def cost_fn (X, y, params): 
    y_pred = pred_val(X, params)
    
    return np.sum(-(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)))
    
    
def pred_val(X, params):
    return sigmoid(np.matmul(X, params))

    
def sigmoid(x):
    return np.exp(x)/(np.exp(x)+1)
    
    
def gradient_descent(X, y, params, learning_rate,num_iters):
    
    cost_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        
        y_pred = pred_val(X, params)
        
        #dimensions of X = [num_samples, num_features]
        #final dimensions have to be = [num_features, 1]
        #transpose of X = [num_features, num_samples]
        #dimensions of y_pred - y = [num_samples, 1]
        #therefore need to premultiply the transpose of X with (y_pred - y)
        
        derivative_of_cost = np.matmul(X.transpose(), y_pred - y)
        derivative_of_cost *= learning_rate
        derivative_of_cost /= len(y) #aka number of samples
        
        params = params - derivative_of_cost
        
        cost_history[i] = cost_fn(X, y, params)
    return (cost_history, params)


'''
3 classes:
    #   Iris-setosa
    #   Iris-versicolor
    #   Iris-virginica

However, we'll be performing binary classification. The class Iris-setosa is linearly separable from the others, so we'll stick to that. 

'''    

df['Class'] = df['Class'].replace(['Iris-setosa'],'0')

df['Class'] = df['Class'].replace(['Iris-virginica','Iris-versicolor'], '1')        

X=df.drop('Class', axis =1)
y=df['Class']

num_samples = len(y)
X = np.hstack((np.ones((num_samples,1)),X))
y = np.array(y, dtype='float64')
y = y[:, np.newaxis]
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

num_iterations = 1000
learning_rate = 0.05        #this is also called alpha

params = np.zeros((X_train.shape[1],1))

print('Inital cost', cost_fn(X_train, y_train, params))
(cost_history, best_params) = gradient_descent(X_train, y_train, params, learning_rate, num_iterations)

print('Final cost', cost_fn(X_train, y_train, best_params))


plt.plot(range(len(cost_history)), cost_history, 'r')       #This graphs the cost vs the iteration number, helps us track when to stop training the model and how fast it's converging
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()

y_test_pred = np.round(pred_val(X_test, best_params))   #rounding the sigmoid results to either 0 or 1. 


from sklearn.metrics import confusion_matrix, classification_report

print("Logistc Regression Testing\n\n")
print(confusion_matrix(y_test, y_test_pred))

print(classification_report(y_test, y_test_pred))
print("\n\n")

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:39:33 2020

@author: Ashay
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('car_data.csv')


def loss_fn(X, y, params):      #mean squared error

    pred_val = np.matmul(X,params)
    
    return np.sum((y-pred_val)**2)/len(y)   # (Summation of (y - y_pred)^2)/number of samples
    

def gradient_descent(X, y, params, learning_rate, iterations):
    
    
    #grad descent means weights = weights - learning_rate * diff(COST FUNCTION)
    #diff of MSE can be simplified to summation((pred_val - truth_val)*X)/number of samples
    #dimensions of X: [samples*features]
    #dimensions of params: [features * 1]
    #dimensions of (pred_val - truth_val) [samples*1]
    #to update params, final dimension has to be [features * 1]
    
    #So, X has to be multiplied to [samples*1] and result has to be [features * 1]
    # only way to do that is to premultiply X transpose [features * samples] with the above
    
    cost_history = np.zeros((iterations,1))
    
    for i in range(iterations):
        pred_val = predict(X, params)
        num_samples = len(y)
        params = params - learning_rate*np.matmul(X.transpose(), pred_val - y)/num_samples
        cost_history[i] = loss_fn(X, y,params)
    
    return (cost_history, params)
        
        

def predict(X, params):
    return np.matmul(X, params)



df['Age'] = 2020-df['Year']
df=df.drop('Year', axis =1)


df = pd.concat([df, pd.get_dummies(df['Car_Name'],prefix='Car_Name')], axis=1)
df = df.drop('Car_Name', axis = 1)

df = pd.concat([df, pd.get_dummies(df['Fuel_Type'],prefix='Fuel_Type')], axis=1)
df = df.drop('Fuel_Type', axis = 1)

df = pd.concat([df, pd.get_dummies(df['Seller_Type'],prefix='Seller_Type')], axis=1)
df = df.drop('Seller_Type', axis = 1)

df = pd.concat([df, pd.get_dummies(df['Transmission'],prefix='Transmission')], axis=1)
df = df.drop('Transmission', axis = 1)

df = pd.concat([df, pd.get_dummies(df['Owner'],prefix='Owner')], axis=1)
df = df.drop('Owner', axis = 1)

y_mean = df['Selling_Price'].mean()
y_std = df['Selling_Price'].std()

dtypes = list(zip(df.dtypes.index, map(str, df.dtypes)))
# Normalize numeric columns.
for column, dtype in dtypes:
    if dtype == 'float64' or dtype == 'int64':
        df[column] -= df[column].mean()
        df[column] /= df[column].std()


X=df.drop('Selling_Price', axis =1) 
y=np.array(df['Selling_Price'])[:, np.newaxis]


n_samples=len(y)

X = np.hstack((np.ones((n_samples, 1)),X))

n_features = np.size(X, 1)

params = np.zeros((n_features, 1))


X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
n_iters = 100
learning_rate = 0.03


(cost_history, best_params) = gradient_descent(X, y, params, learning_rate, n_iters)


plt.plot(range(len(cost_history)), cost_history, 'r')


plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.show()

y_pred = predict(X_test, best_params)

y_pred *= y_std
y_pred += y_mean

y_test *= y_std
y_test += y_mean

plt.plot(y_pred,'r')
plt.plot(y_test,'b')
plt.title('Predicted and Truth values for test set')
plt.legend(['Predicted values','Real Values'])
plt.show()    
    
    
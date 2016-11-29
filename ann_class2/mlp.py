'''
mlp.py --- simple multi layer perceptron
'''

import numpy as np

def forward(X, W1, b1, W2, b2):

    Z = X.dot(W1) + b1
    # Z = 1 / 1 + np.exp(-Z) # sigmoid
    Z[Z < 0] = 0 # relu

    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)

    return Y, Z

def derivative_w2(Z, T, Y):
    return Z.T.dot(Y - T)

def derivative_b2(T, Y):
    return (Y- T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    # return X.T.dot(((Y-T).dot(W2.T) * (Z*(1-Z)))) # sigmoid
    return X.T.dot(((Y_T).dot(W2.T) * np.sign(Z))) # relu

def derivative_b1(Z, T, Y, W2):
    # return ((Y-T).dot(W2.T) * (Z*(1-Z))).sum(axis=0) # sigmoid
    return ((Y-T).dot(W2.T) * np.sign(Z)).sum(axis=0) # relu

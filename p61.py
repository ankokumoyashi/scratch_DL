import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def identity_function(x):
    return x

def forward(x):
    A1 = np.dot(X,W1) +B1

    Z1 = sigmoid(A1)

    A2 = np.dot(Z1,W2) + B2
    Z2 = sigmoid(A2)

    A3 = np.dot(Z2,W3) + B3
    #Z3 = sigmoid(A3)
    Z3 = softmax(A3)
    Y = identity_function(Z3)
    return Y

def softmax(a):
    c = np.max(a)
    print("expまえ",a)
    exp_a = np.exp(a-c)
    print("expあと",exp_a)
    exp_a_sum = np.sum(exp_a)
    print("exp",exp_a_sum)
    return exp_a/exp_a_sum

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
print(forward(X))

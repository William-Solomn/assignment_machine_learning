import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def costFunction(theta, X, y):
    m = X.shape[0]
    0
    z = X.dot(theta)
    J = (-y.T.dot(np.log(sigmoid(z))) - (1 - y.T).dot(np.log(1 - sigmoid(z)))) / m
    return J[0]


def gradient(theta, X, y):
    m = X.shape[0]  # X.shape = (100,3)
    grad = np.zeros_like(theta)
    z = X.dot(theta.reshape(-1, 1))  # theta.shape = (3,) theta.reshape(-1,1).shape = (3,1)
    grad = (1.0 / m) * X.T.dot((sigmoid(z) - y))
    return grad.flatten()


# def mapFeature(X):
#     poly = PolynomialFeatures(6)
#     XX = poly.fit_transform(X)
#     return XX

def mapFeature(x1col, x2col):
    degrees = 6
    out = np.ones((x1col.shape[0], 1))
    for i in range(1, degrees + 1):
        for j in range(0, i + 1):
            term1 = x1col ** (i - j)
            term2 = x2col ** (j)
            term = (term1 * term2).reshape(term1.shape[0], 1)
            out = np.hstack((out, term))
    return out


def costFunctionReg(theta, lamb, X, y):
    m = X.shape[0]
    J = 0
    z = X.dot(theta)
    J = (-y.T.dot(np.log(sigmoid(z))) - (1 - y.T).dot(np.log(1.0 - sigmoid(z)))) / m
    J += lamb / (2.0 * m) * theta[1:].T.dot(theta[1:])
    return J[0]


def gradientReg(theta, lamb, X, y):
    m = X.shape[0]
    np.zeros_like(theta)
    z = X.dot(theta.reshape(-1, 1))
    grad = (1.0 / m) * X.T.dot((sigmoid(z) - y)) + (lamb / m) * theta.reshape(-1, 1)
    grad[0] -= (lamb / m) * theta[0]
    return grad.flatten()

import numpy as np
import random
import math

from bitstring import xrange


class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.ww = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        m = X_batch.shape[0]
        loss = 0
        z = X_batch.dot(self.w)
        loss = (-y_batch.T.dot(np.log(self.sigmoid(z))) - (1.0 - y_batch.T).dot(np.log(1.0 - self.sigmoid(z)))) / m
        grad = np.zeros_like(self.w)
        grad = X_batch.T.dot(self.sigmoid(z) - y_batch) / m
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        return loss, grad

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
              batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            Sample_batch = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[Sample_batch]
            y_batch = y[Sample_batch]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w += -learning_rate * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        y_pred = self.sigmoid(X.dot(self.w))
        for i in range(X.shape[0]):
            if y_pred[i] >= 0.50:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        """

        num_train, dim = X.shape
        self.ww = np.zeros((dim, 10))
        for it in range(10):
            y_train = []
            for label in y:
                if label == it:
                    y_train.append(1)
                else:
                    y_train.append(0)
            y_train = np.array(y_train)
            self.w = None
            print("it = ", it)
            self.train(X, y_train, learning_rate, num_iters, batch_size)
            self.ww[:, it] = self.w

    def one_vs_all_predict(self, X):
        laybels = self.sigmoid(X.dot(self.ww))
        y_pred = np.argmax(laybels, axis=1)
        return y_pred

import numpy as np
from bitstring import xrange
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.ww = None

    def sigmoid(self,z):
        return 1.0 / ( 1.0 + np.exp(-z) )

    def loss(self, X_batch, y_batch):
        
        m = X_batch.shape[0] 

        loss = 0

        z = X_batch.dot(self.w) 

        loss = (-y_batch.T.dot(np.log(self.sigmoid(z))) - (1.0 - y_batch.T).dot(np.log(1.0 - self.sigmoid(z))))/m 

        grad = np.zeros_like(self.w) 
        grad = X_batch.T.dot(self.sigmoid(z) - y_batch) /m 

        return loss,grad

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):
        
        num_train, dim = X.shape 

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            Sample_batch = np.random.choice(np.arange(num_train), batch_size)

            X_batch = X[Sample_batch]
            y_batch = y[Sample_batch]

            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            self.w += -learning_rate*grad

            if verbose and it % 50 == 0:
                print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history



    def predict(self, X):

        y_pred = np.zeros(X.shape[0])

        # 实现此方法。将预测的标签存储在y_pred中。           
        y_pred = self.sigmoid(X.dot(self.w))
        for i in range(X.shape[0]):
            if y_pred[i] >= 0.50:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred


    def deal_all(self, X, y, learning_rate=0.09, num_iters=100,batch_size=200,verbose = False):

        loss_history_one = np.empty(shape=[1,num_iters],dtype=float)# [一行，学习步骤次数 列]
        loss_history_servel = np.empty(shape=[10,num_iters],dtype=float)

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
            print("本次训练的数字是 = ", it)

            loss_history_one[[0],:] = self.train(X, y_train, learning_rate, num_iters, batch_size)
            loss_history_servel[[it],:] = loss_history_one

            self.ww[:, it] = self.w
        return loss_history_servel
        
    def one_vs_all_predict(self, X):
        laybels = self.sigmoid(X.dot(self.ww))
        y_pred = np.argmax(laybels,axis=1)
        return y_pred
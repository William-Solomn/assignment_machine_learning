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
        """
        计算损失函数及其导数。
        子类将覆盖它。
        输入:
        - X_batch:形状为(N,D)的numpy数组,包含N的minibatch
        数据点;每个点都有维数d。
        - y_batch:shape(N,)的numpy数组,包含minibatch的标签。
        返回:一个元组,包含:
        -作为单一浮动的损失
        -相对于自身的梯度。w;与W形状相同的数组
        """

        # 计算损失和导数
        m = X_batch.shape[0]
        loss = 0
        z = X_batch.dot(self.w)
        loss = (-y_batch.T.dot(np.log(self.sigmoid(z))) - (1.0 - y_batch.T).dot(np.log(1.0 - self.sigmoid(z))))/m
        grad = np.zeros_like(self.w)
        grad = X_batch.T.dot(self.sigmoid(z) - y_batch) /m

        return loss,grad

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        使用随机梯度下降训练这个线性分类器。
        Inputs:
        - X:包含训练数据的shape (N,D)的numpy数组;有N个d维的训练样本。
        - y:包含训练标签的shape (N,)的numpy数组;
        - learning_rate: (float)优化的学习率。
        - num_iters: (integer)优化时要采取的步骤数
        - batch_size:(整数)每个步骤中要使用的训练示例的数量。
        - verbose: (boolean)如果为true,则在优化过程中打印进度。
        产出:
        包含每次训练迭代的损失函数值的列表。
        """
        num_train, dim = X.shape
        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)
        loss_history = []
        for it in xrange(num_iters):
            X_batch = None
            y_batch = None
            '''
            # TODO: #
            #来自训练数据的样本batch_size元素及其#
            #在这一轮梯度下降中使用的相应标签。#
            #将数据存储在X_batch中.并将它们对应的标签存储在#
            # y _ batch采样后.X_batch应具有形状(batch_size.dim) #
            #和y_batch应具有shape (batch_size.)#
            #提示:使用np.random.choice生成索引。用#采样
            #替换比不替换的采样更快。
            '''
            
            Sample_batch = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[Sample_batch]
            y_batch = y[Sample_batch]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            if verbose and it % 149 == 0:
                # print ('iteration %d / %d: 损失情况： %f' % (it, num_iters, loss))
                print ('损失情况是： %f' % (loss))

        return loss_history

    def predict(self, X):
        """
            使用此线性分类器的训练权重来预测数据点的标签。
            输入:
            训练数据的X: D x N数组。每一列都是一个D维点。
            返回
            -y_pred:x中数据的预测标签。y _ pred是一维的
            长度为N的数组,每个元素都是给出预测类的整数。
        """
        y_pred = np.zeros(X.shape[0])

        # 实现此方法。将预测的标签存储在y_pred中。           
        y_pred = self.sigmoid(X.dot(self.w))
        for i in range(X.shape[0]):
            if y_pred[i] >= 0.50:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred


    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):
    #     """
    #     使用随机梯度下降训练这个线性分类器。
    #     输入:
    #     - X:包含训练数据的shape (N,D)的numpy数组;有N个
    #     d维的训练样本。
    #     - y:包含训练标签的shape (N,)的numpy数组;
    #     - learning_rate: (float)优化的学习率。
    #     - num_iters: (integer)优化时要采取的步骤数
    #     - batch_size:(整数)每个步骤中要使用的训练示例的数量。
    #     - verbose: (boolean)如果为true,则在优化过程中打印进度。
    #     """
        loss_history_one = np.empty(shape=[1,num_iters],dtype=float)
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
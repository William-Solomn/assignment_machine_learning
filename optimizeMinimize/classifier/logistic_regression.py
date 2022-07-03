import locale
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.optimize as op

class LogisticRegression(object):

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def costFunction(self, theta, X, y):
        m = X.shape[0]
        J = 0
        z = X.dot(theta)
        J = (-y.T.dot(np.log(self.sigmoid(z))) - (1 - y.T).dot(np.log(1 - self.sigmoid(z)))) / m
        return J[0]

    def gradient(self, theta, X, y): 
        m = X.shape[0] #X.shape (100,3)
        grad = np.zeros_like(theta)
        z = X.dot(theta.reshape(-1,1)) # theta.shape = (3,) theta.reshape(-1,1).shape = (3,1)
        grad = (1.0 / m) * X.T.dot(( self.sigmoid(z) - y ))  
        return grad.flatten()


    def mapFeature(self, x1col, x2col):
        degrees = 6
        out = np.ones((x1col.shape[0],1))
        for i in range(1, degrees+1):
            for j in range(0, i+1):
                term1 = x1col ** (i-j)
                term2 = x2col ** (j)
                term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
                out   = np.hstack(( out, term ))
        return out

    def costFunctionReg(self, theta, lamb, X, y):
        m = X.shape[0]
        J = 0
        z = X.dot(theta)
        J = (-y.T.dot(np.log(self.sigmoid(z))) - (1 - y.T).dot(np.log(1.0 - self.sigmoid(z)))) / m
        J += lamb/(2.0*m) * theta[1:].T.dot(theta[1:]) # 公式换算，返回第一个属性值
        return J[0]

    def gradientReg(self, theta, lamb, X, y):
        m = X.shape[0]
        grad = np.zeros_like(theta)
        z = X.dot(theta.reshape(-1,1))
        grad = (1.0 / m) * X.T.dot((self.sigmoid(z) - y)) + (lamb / m) * theta.reshape(-1,1)
        grad[0] -= (lamb/m) * theta[0]
        return grad.flatten()



    def loadData(self, path):
        path = os.getcwd() + path # 获取到当前工作目录
        data = np.loadtxt(path, dtype=float, delimiter=',')
        X = data[:, 0:2] # 点 根据数据集的不同而不同
        y = data[:, 2] # 已经分类好的label
        return X, y

    def plotScatter(self, path):
        X, y = self.loadData(path)
        y = y.astype(int) # 确保label是int型的0或1
        neg = (y == 0)
        pos = (y == 1)
        print(type(neg))
        lable1 = plt.scatter(X[neg, 0], X[neg, 1], marker='^', c='g', alpha=0.7) # 绘制散点图，先通过上面的两行分类成neg和pos，然后再
        lable2 = plt.scatter(X[pos, 0], X[pos, 1], marker='v', c='r', alpha=0.7)
        plt.legend((lable1, lable2), ('pass', 'not pass'))
        plt.xlabel('exam 1')
        plt.ylabel('exam 2')

    def decisionBoundary(self, theta, path):
        self.plotScatter(path)
        X, y = self.loadData(path)
        y.astype(int)
        boundary_xs = np.array([np.min(X[:, 0]), np.max(X[:, 0])]) # 该参数为第一列数据的[最小值,最大值]的数组
        boundary_ys = (-1. / theta[2]) * (theta[0] + theta[1] * boundary_xs)
        plt.plot(boundary_xs, boundary_ys)

    def decisionBoundaryReg(self, theta, X, y, lamb=0.):
        theta, mincost = self.optimizeRegularizedTheta(theta, X, y, lamb)
        xvals = np.linspace(-1, 1.5, 50)
        yvals = np.linspace(-1, 1.5, 50)
        zvals = np.zeros((len(xvals), len(yvals)))
        for i in range(len(xvals)):
            for j in range(len(yvals)):
                myfeaturesij = self.mapFeature(np.array([xvals[i]]), np.array([yvals[j]]))
                zvals[i][j] = np.dot(theta, myfeaturesij.T)
        zvals = zvals.transpose()

        u, _ = np.meshgrid(xvals, yvals)
        plt.contour(xvals, yvals, zvals, [0])
        plt.title("Decision Boundary with Lambda = %d" % lamb)

    def optimizeRegularizedTheta(self, theta, X, y, lamb=0.):
        result = op.minimize(self.costFunctionReg, theta, args=(lamb, X, y), method='BFGS',
                            options={"maxiter": 500, "disp": False})
        return np.array([result.x]), result.fun

    def subplotDecisionBoundary(self, path, theta, X, y):
        plt.figure(figsize=(12, 10))
        plt.subplot(221)
        self.plotScatter(path)
        self.decisionBoundaryReg(theta, X, y, 0.)
        plt.subplot(222)
        self.plotScatter(path)
        self.decisionBoundaryReg(theta, X, y, 1.)
        plt.subplot(223)
        self.plotScatter(path)
        self.decisionBoundaryReg(theta, X, y, 10.)
        plt.subplot(224)
        self.plotScatter(path)
        self.decisionBoundaryReg(theta, X, y, 100.)


    def LogisticReg(self, path):
        X, y = self.loadData(path)
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]  # add one col
        y = np.c_[y]
        X = self.mapFeature(X[:, 1], X[:, 2])
        initial_theta = np.zeros(X.shape[1])
        self.subplotDecisionBoundary(path, initial_theta, X, y)
        plt.show()
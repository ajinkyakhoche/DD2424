'''
Deep Learning for Data Science: Assignment 3
Submitted by: Ajinkya Khoche (khoche@kth.se)
Description -   Generalize to k-layer Neural Network
            -   Implement Batch Normalization
            -   Implement moving average
            -   Training using mini batch Gradient Descent
'''
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from scipy.spatial import distance
from sklearn import preprocessing
import copy
import os
import pickle

class kLayerNN(object):
    def __init__(self, filePath, GradCheckParams, GradDescentParams, NNParams):
        if NNParams['loadAllBatches']:
            # Call LoadBatch function to get training, validation and test set data
            X1, Y1, y1 = self.LoadBatch(
                filePath + '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_1')
            X2, Y2, y2 = self.LoadBatch(
                filePath + '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_2')
            X3, Y3, y3 = self.LoadBatch(
                filePath + '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_3')
            X4, Y4, y4 = self.LoadBatch(
                filePath + '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_4')
            X5, Y5, y5 = self.LoadBatch(
                filePath + '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_5')
            X6, Y6, y6 = self.LoadBatch(
                filePath + '/Datasets/cifar-10-python/cifar-10-batches-py/test_batch')

            self.Xtrain = np.concatenate((X1, X2, X3, X4, X5[:, 0:9000]), axis=1)
            self.Ytrain = np.concatenate((Y1, Y2, Y3, Y4, Y5[:, 0:9000]), axis=1)
            self.ytrain = np.concatenate((y1, y2, y3, y4, y5[0:9000]))
            self.Xval = X5[:, 9000:10000]
            self.Yval = Y5[:, 9000:10000]
            self.yval = y5[9000:10000]
            self.Xtest = X6
            self.Ytest = Y6
            self.ytest = y6
        else:
            # Call LoadBatch function to get training, validation and test set data
            self.Xtrain, self.Ytrain, self.ytrain = self.LoadBatch(filePath +
                                                                '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_1')
            self.Xval, self.Yval, self.yval = self.LoadBatch(filePath +
                                                            '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_2')
            self.Xtest, self.Ytest, self.ytest = self.LoadBatch(filePath +
                                                                '/Datasets/cifar-10-python/cifar-10-batches-py/test_batch')

        # Normalize Data by subtracting mean
        self.ZeroMean()

        # Assign all GradCheckParams
        self.h = GradCheckParams['h']
        self.eps = GradCheckParams['eps']
        self.tol1 = GradCheckParams['tol1']

        # Assign all GradDescentParams
        self.sigma = GradDescentParams['sigma']
        self.eta = GradDescentParams['eta']
        self.lmbda = GradDescentParams['lmbda']
        self.rho = GradDescentParams['rho']
        self.nEpoch = GradDescentParams['nEpoch']
        self.nBatch = GradDescentParams['nBatch']
        #self.BatchSize = round(self.Xtrain.shape[1]/self.nBatch)
        self.epsilon = GradDescentParams['epsilon']
        self.alpha = GradDescentParams['alpha']

        # Assign all NNParams
        self.d = NNParams['d']
        self.k = NNParams['k']
        self.n = NNParams['n']
        self.m = NNParams['m']
        self.nLayers = len(self.m) + 1
        self.batchNorm = NNParams['batchNorm']

        # Initialize Weights
        self.InitializeWeightAndBias('Gaussian')

        # Initialize mu_avg and var_avg for exponential moving average
        self.mu_avg = [np.zeros_like(self.b[i]) for i in range(1,self.nLayers)]
        self.var_avg = [np.zeros_like(self.b[i]) for i in range(1,self.nLayers)]  

    def unpickle(self, file):
        '''
        Function: unpickle
        Input: file name
        Output: data in form of dictionary
        '''
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def LoadBatch(self, fileName):
        '''
        Function: LoadBatch
        Input: path to a file
        Output: Images (X), labels (y) and one-hot encoding (Y)
        '''
        dict = self.unpickle(fileName)
        X = np.array(dict[b'data']/255)
        y = np.array(dict[b'labels'])
        binarizer = preprocessing.LabelBinarizer()
        binarizer.fit(range(max(y.astype(int)) + 1))
        Y1 = np.array(binarizer.transform(y.astype(int))).T
        return np.transpose(X), np.transpose(Y1.T), y

    def ZeroMean(self):
        mean_Xtrain = np.reshape(np.mean(self.Xtrain, 1), (-1, 1))
        self.Xtrain -= mean_Xtrain
        self.Xval -= mean_Xtrain
        self.Xtest -= mean_Xtrain

    def InitializeWeightAndBias(self, type='Gaussian'):
        '''
        Input: Type of weights. Possible choices: Gaussian, Javier, He
        Output: W and b; both are lists
        '''
        if type == 'Gaussian':
            np.random.seed(400)
            self.W = []
            self.W.append([])
            self.b = []
            self.b.append([])

            self.W.append(np.random.randn(
                list(self.m)[0], self.d) * self.sigma)
            self.b.append(np.zeros((list(self.m)[0], 1)))

            for i in range(self.nLayers - 2):
                self.W.append(np.random.randn(
                    self.m[i+1], self.m[i]) * self.sigma)
                self.b.append(np.zeros((self.m[i+1], 1)))

            self.W.append(np.random.randn(
                self.k, list(self.m)[-1]) * self.sigma)
            self.b.append(np.zeros((self.k, 1)))
        # FUTURE: Add other initializations

    def BatchNormalize(self, s, mu, var):
        V = np.array([var + self.epsilon])
        Vinv0_5 = V**-0.5
        sHat = np.multiply((s-mu), Vinv0_5.T) 
        return sHat

    def BatchNormBackPass(self, dJdsHat, s, mu, var):
        '''
        Input: g (dJ/dsHat), s, mu, var
        Output: g (dJ/ds)
        Comments: Refer to last slide of Lec 4
        '''
        N = dJdsHat.shape[0]
        V = np.array([var + self.epsilon])
        Vinv1_5 = V**-1.5
        dJdvar = -0.5 * np.sum(np.multiply(np.multiply(dJdsHat, Vinv1_5),(s-mu).T), axis = 0)

        Vinv0_5 = V**-0.5
        dJdmu = - np.sum(np.multiply(dJdsHat, Vinv0_5), axis = 0)

        dJds = np.multiply(dJdsHat, Vinv0_5) + 2/N * np.multiply(dJdvar, (s-mu).T) + dJdmu/N

        return dJds      


    def EvaluateClassifier2(self, x, Wt, bias):
        N = x.shape[1]
        if x.ndim == 1:
            x = np.reshape(x, (-1, 1))
        h = []
        s = []
        sHat = []
        mu = []
        var = []
        mu.append([])
        var.append([])
        h.append(x)
        s.append([])
        sHat.append([])
    

        for i in range(self.nLayers-1):
            s.append(np.dot(Wt[i+1], h[i]) + bias[i+1])
            # calculate mu and variance
            mu.append(np.reshape(np.mean(s[i+1], axis = 1), (-1, 1)))
            # var.append(np.reshape(np.sum(((s[i+1]-mu[i+1])**2), 1)/N, (-1, 1))) #DIAG OF THIS IS SCALAR!!!
            # DIAG OF THIS IS SQUARE MATRIX!!!
            var.append(np.sum(((s[i+1]-mu[i+1])**2), 1)/N)

            # Exponential Moving Average
            # temp_var = 0
            # for j in range(self.nLayers):
            #     if self.mu_avg[j].all() == 0:
            #         temp_var = temp_var + 1

            if self.mu_avg[i].all() == 0:
                # all elements are zero, so this is first ever evaluation step
                self.mu_avg[i] = mu[i+1]
                self.var_avg[i] = var[i+1]
            else:    
                self.mu_avg[i] = self.alpha * self.mu_avg[i] + (1 - self.alpha) * mu[i+1]
                self.var_avg[i] = self.alpha * self.var_avg[i] + (1 - self.alpha) * var[i+1]

            sHat.append(self.BatchNormalize(s[i+1], mu[i+1], var[i+1]))
            if self.batchNorm:
                h.append(np.maximum(0, sHat[i+1]))     ###CHANGE TO sHat
            else:
                h.append(np.maximum(0, s[i+1]))

        # for final layer:
        s.append(np.dot(Wt[self.nLayers],
                        h[self.nLayers-1]) + bias[self.nLayers])
        # compute softmax function of s
        p = np.exp(s[self.nLayers])
        p = p / np.sum(p, axis=0)
        return p, s, sHat, h, mu, var

    def ComputeCost2(self, X, Y, Wt, bias):
        N = X.shape[1]
        p, _, _, _, _, _ = self.EvaluateClassifier2(X, Wt, bias)
        A = np.diag(np.dot(Y.T, p))
        B = -np.log(A)
        C = np.sum(B)
        Sum = 0
        for i in range(self.nLayers):
            Sum += np.sum(np.square(Wt[i]))
        #D = self.lmbda * (np.sum(np.square(Wt[0]))+np.sum(np.square(Wt[1])))
        D = self.lmbda * Sum

        J = C/N + D
        # J = (np.sum(-np.log(np.dot(Y.T, p))))/M + lmbda * np.sum(np.square(W))
        return J

    def ComputeAccuracy2(self, X, y, Wt, bias):
        N = X.shape[1]
        p, _, _, _, _, _ = self.EvaluateClassifier2(X, Wt, bias)
        k = np.argmax(p, axis=0)
        a = np.where((k.T - y) == 0, 1, 0)
        acc = sum(a)/N
        return acc

    def PerformanceEval(self, grad_W, grad_b, numgrad_W, numgrad_b):
        '''Method 2: Check absolute difference'''
        # Layer 1
        abs_W = []
        abs_b = []
        for i in range(self.nLayers):
            abs_W.append(abs(np.array(grad_W[i+1]) - np.array(numgrad_W[i+1])))
            abs_b.append(abs(np.array(grad_b[i+1]) - np.array(numgrad_b[i+1])))

        '''Method 1: check if relative error is small'''
        # Layer1
        rel_W = []
        rel_b = []

        for i in range(self.nLayers):
            rel_W.append(np.divide(abs(grad_W[i+1] - numgrad_W[i+1]),
                                   max(self.eps, (abs(grad_W[i+1]) + abs(numgrad_W[i+1])).all())))
            rel_b.append(np.divide(abs(grad_b[i+1] - numgrad_b[i+1]),
                                   max(self.eps, (abs(grad_b[i+1]) + abs(numgrad_b[i+1])).all())))
        
        avg_abs_W = np.zeros(self.nLayers)
        avg_abs_b = np.zeros(self.nLayers)
        avg_rel_W = np.zeros(self.nLayers)
        avg_rel_b = np.zeros(self.nLayers)

        for k in range(self.nLayers):
            avg_abs_W[k] = np.mean(abs_W[k])
            avg_abs_b[k] = np.mean(abs_b[k])
            avg_rel_W[k] = np.mean(rel_W[k])
            avg_rel_b[k] = np.mean(rel_b[k])

        return np.mean(avg_abs_W), np.mean(avg_abs_b), np.mean(avg_rel_W), np.mean(avg_rel_b)

    def ComputeGradients(self, X, Y, Wt, bias):
        N = X.shape[1]
        grad_W = []
        grad_W.append([])
        grad_b = []
        grad_b.append([])
        grad_W.append(np.zeros((list(self.m)[0], Wt[1].shape[1])))
        grad_b.append(np.zeros((list(self.m)[0], 1)))

        for i in range(self.nLayers - 2):
            grad_W.append(np.zeros((list(self.m)[i], list(self.m)[i+1])))
            grad_b.append(np.zeros((list(self.m)[i+1], 1)))

        grad_W.append(np.zeros((self.k, list(self.m)[-1])))
        grad_b.append(np.zeros((self.k, 1)))

        # Forward Pass
        p, s, sHat, h, mu, var = self.EvaluateClassifier2(X, Wt, bias)

        # Backward Pass
        # gradients for last layer
        g = - (Y - p).T
        grad_b[self.nLayers] = np.sum(g.T, 1)/N
        grad_b[self.nLayers] = np.reshape(grad_b[self.nLayers], (-1, 1))
        grad_W[self.nLayers] = (
            np.dot(g.T, h[self.nLayers-1].T))/N + 2 * self.lmbda * Wt[self.nLayers]
        # Propogate gradient vector gi to previous layer:
        g = np.dot(g, Wt[self.nLayers])
        if self.batchNorm:
            ### CHANGE s to sHat
            sTemp = np.where(sHat[self.nLayers-1] > 0, 1, 0)
        else:
            sTemp = np.where(s[self.nLayers-1] > 0, 1, 0)

        '''
        THERE IS AN ERROR IN ASSIGNMENT NOTES: ref eq 22 on page 4.
        instead of 
            g_i * diag(Ind(ŝ_i(k-1)> 0))
        it was
            g_i * Ind(ŝ_i(k-1)> 0)
        which finally gave convergence of analytical and numerical 
        gradients. I wasted at least 50 hours on finding out this bug!!
        '''
        # if g.shape[0] < self.m[self.nLayers-2]: 
        #     g = np.multiply(g, np.reshape(np.diag(sTemp), (-1, 1)))
        # else:
        #     g = np.multiply(g, np.diag(sTemp))
        if g.shape[0] < self.m[self.nLayers-2]: 
            g = np.multiply(g, np.reshape(np.diag(sTemp), (-1, 1)))
        else:
            g = np.multiply(g, sTemp.T)

        for i in range(self.nLayers-1, 0, -1):
            if self.batchNorm:
                ### UNCOMMENT THIS LINE IF NOT BATCH NORM
                g = self.BatchNormBackPass(g, s[i], mu[i], var[i])

            grad_b[i] = np.sum(g.T, 1)/N
            grad_b[i] = np.reshape(grad_b[i], (-1, 1))
            grad_W[i] = (np.dot(g.T, h[i-1].T))/N + 2 * self.lmbda * Wt[i]
            # Propagate gradient vector to previous layer (if i > 1):
            if i > 1:
                g = np.dot(g, Wt[i])
                ### CHANGE s to sHat
                if self.batchNorm:
                    sTemp = np.where(sHat[i-1] > 0, 1, 0)
                else:
                    sTemp = np.where(s[i-1] > 0, 1, 0)
                
                # if g.shape[0] < self.m[i-2]:
                #     g = np.multiply(g, np.reshape(np.diag(sTemp), (-1,1)))
                # else:
                #     g = np.multiply(g, np.diag(sTemp))
                if g.shape[0] < self.m[i-2]:
                    g = np.multiply(g, np.reshape(np.diag(sTemp), (-1,1)))
                else:
                    g = np.multiply(g, sTemp.T)

        return grad_W, grad_b

    def ComputeGradientsNumSlow(self, X, Y, Wt, bias):
        numgrad_W = []
        numgrad_W.append([])
        numgrad_b = []
        numgrad_b.append([])
        numgrad_W.append(np.zeros((list(self.m)[0], Wt[1].shape[1])))
        numgrad_b.append(np.zeros((list(self.m)[0], 1)))

        for i in range(self.nLayers - 2):
            numgrad_W.append(np.zeros((list(self.m)[i+1], list(self.m)[i])))
            numgrad_b.append(np.zeros((list(self.m)[i+1], 1)))

        numgrad_W.append(np.zeros((self.k, list(self.m)[-1])))
        numgrad_b.append(np.zeros((self.k, 1)))

        for j in range(len(bias)):
            # numgrad_b[j] = np.zeros(b[j].shape)
            for i in range(len(bias[j])):
                b_try = copy.deepcopy(bias)
                b_try[j][i] -= self.h
                c1 = self.ComputeCost2(X, Y, Wt, b_try)

                b_try = copy.deepcopy(bias)
                b_try[j][i] += self.h
                c2 = self.ComputeCost2(X, Y, Wt, b_try)

                numgrad_b[j][i] = (c2 - c1)/(2*self.h)

        for j in range(len(Wt)):
            # numgrad_W = np.zeros(W[j].shape)
            for i in range(len(Wt[j])):
                for k in range(len(Wt[j][0])):
                    W_try = copy.deepcopy(Wt)
                    W_try[j][i][k] -= self.h
                    c1 = self.ComputeCost2(X, Y, W_try, bias)

                    W_try = copy.deepcopy(Wt)
                    W_try[j][i][k] += self.h
                    c2 = self.ComputeCost2(X, Y, W_try, bias)

                    numgrad_W[j][i][k] = (c2-c1)/(2*self.h)

        return numgrad_W, numgrad_b

    def CheckGradients(self, X, Y, Wt, bias, mode='slow'):

        [grad_W, grad_b] = self.ComputeGradients(X, Y, Wt, bias)

        if mode == 'fast':
            [numgrad_W, numgrad_b] = self.ComputeGradientsNum(X, Y, Wt, bias)
        elif mode == 'slow':
            [numgrad_W, numgrad_b] = self.ComputeGradientsNumSlow(
                X, Y, Wt, bias)

        # Check Performance
        abs_W, abs_b, rel_W, rel_b = self.PerformanceEval(grad_W, grad_b, numgrad_W, numgrad_b)
        return abs_W, abs_b, rel_W, rel_b 

    def plotLoss(self, Jtrain, Jval, nEpoch):
        iterations = list(range(1, nEpoch + 1))
        plt.figure()
        plt.xlim((0,self.nEpoch+1))
        plt.ylim((1,2.5))
        plt.plot(iterations, Jtrain, linewidth=3, label='Training Loss')
        plt.plot(iterations, Jval, linewidth=3, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.xlabel('No. of Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Loss')
        plt.grid()

    def MiniBatchGD(self, Xtrain, Ytrain, Xval, Yval, Wt, bias, PlotLoss):
        # Initialize Wstar and bstar (these would serve as updated weights)
        Wstar = Wt
        bstar = bias

        # Initialize Momentum Vectors
        v_b = [np.zeros_like(a) for a in bstar]
        v_W = [np.zeros_like(a) for a in Wstar]

        JtrainList = np.zeros(self.nEpoch)
        JvalList = np.zeros(self.nEpoch)
        # No. of batch iterations
        nBatchIter = int(Xtrain.shape[1]/self.nBatch)
        for i in range(self.nEpoch):
            for j in range(nBatchIter):
                # extract a batch for training
                j_start = j * self.nBatch
                j_end = (j+1)*self.nBatch
                Xbatch = Xtrain[:, j_start: j_end]
                Ybatch = Ytrain[:, j_start: j_end]

                # Forward pass and Back Propagation:
                [grad_W, grad_b] = self.ComputeGradients(
                    Xbatch, Ybatch, Wstar, bstar)

                for k in range(len(grad_W)-1):
                    # Momentum Update:
                    v_b[k+1] = self.rho * v_b[k+1] + self.eta * grad_b[k+1]
                    v_W[k+1] = self.rho * v_W[k+1] + self.eta * grad_W[k+1]
                    # Weight/bias update:
                    # HOW IT WAS BEFORE MOMENTUM--
                    # Wstar[k] = Wstar[k] - GDparams.eta * grad_W[k]
                    # bstar[k] = bstar[k] - GDparams.eta * grad_b[k]
                    # HOW IT IS WITH MOMENTUM--
                    Wstar[k+1] = Wstar[k+1] - v_W[k+1]
                    bstar[k+1] = bstar[k+1] - v_b[k+1]

            Jtrain = self.ComputeCost2(Xtrain, Ytrain, Wstar, bstar)
            Jval = self.ComputeCost2(Xval, Yval, Wstar, bstar)
            print('Epoch ' + str(i) + '- Training Error = ' +
                  str(Jtrain) + ', Validation Error = ' + str(Jval))
            JtrainList[i] = Jtrain
            JvalList[i] = Jval
        if PlotLoss == True:
            self.plotLoss(JtrainList, JvalList, self.nEpoch)
        return Wstar, bstar

    def CoarseToFineRandomSearch(self, eta_range, lmbda_range, no_of_epochs, nIter):
        # Modify no. of epochs
        self.nEpoch = no_of_epochs
        # Create Text file to store data
        fp = open('parameterTest6.txt', 'w')
        text1 = 'Coarse to Fine parameter search test:\neta_range: 1e'+str(eta_range[0]) + '- 1e'+str(eta_range[1])+'\nlmbda range: 1e'+str(lmbda_range[0]) + '- 1e'+str(lmbda_range[1])+'\nnEpochs = ' + \
            str(no_of_epochs) + '\nnIter: ' + str(nIter) + '\n\n'
        fp.write(text1)
        fp.write('eta\t\teta_exponent\tlmbda\t\tlmbda_exponent\tAccuracy(%)\n')
        for i in range(nIter):
            # Generate random eta
            e = eta_range[0] + (eta_range[1] - eta_range[0]
                                )*np.random.rand(1, 1)
            self.eta = 10**e

            # Generate random lmbda
            f = lmbda_range[0] + (lmbda_range[1] - lmbda_range[0]
                                  )*np.random.rand(1, 1)
            self.lmbda = 10**f

            # Gradient Descent
            [Wstar, bstar] = self.MiniBatchGD(
                self.Xtrain, self.Ytrain, self.Xval, self.Yval, self.W, self.b, False)

            # Compute Accuracy on Validation set
            acc = self.ComputeAccuracy2(self.Xtest, self.ytest, Wstar, bstar)
            print('>>>>ETA ' + str(self.eta) + ':' +
                  'LAMBDA: ' + str(self.lmbda) + '<<<<')
            print('>>>>ITERATION ' + str(i) + ':' +
                  'ACCURACY: ' + str(round(acc*100, 2)) + '%<<<<')
            # Write parameters to text file
            fp.write(str(self.eta) + '\t' + str(e) + '\t' +
                     str(self.lmbda) + '\t' + str(f) + '\t' + str(round(acc*100, 2)) + '\n')
            acc = 0
        print('The results of Coarse to Fine search can be found in :' + fp.name + ' file in root folder' )
        fp.close()
        return

    def FinalTest(self):
        # Gradient Descent
        [Wstar, bstar] = self.MiniBatchGD(
            self.Xtrain, self.Ytrain, self.Xval, self.Yval, self.W, self.b, True)

        # Compute Accuracy on Validation set
        acc = self.ComputeAccuracy2(self.Xtest, self.ytest, Wstar, bstar)

        print('Accuracy obtained on test set is: ' + str(round(acc*100, 2)) + '%')


def main():
    '''
    Specify path to Datasets in variable 'filePath'
    '''
    filePath = os.getcwd()
    '''
    Specify Gradient Checking parameters in 'GradCheckParams'

    h:      Step Size
    eps:    epsilon for placing in denominator of relative gradient checking
    tol1:   Tolerance for absolute gradient checking
    '''
    GradCheckParams = {'h': 1e-6, 'eps': 1e-8, 'tol1': 1e-7}

    '''
    Specify Gradient Descent parameters in 'GradDescentParams'

    sigma:  Variance to initialize random weights
    eta:    Learning Rate
    lmbda:  Regularization Parameter
    rho:    Momentum
    nEpoch: No. of epochs
    nBatch: Batch size
    epsilon:small number used for division in Batch Normalization function
    '''
    GradDescentParams = {'sigma': 0.001, 'eta': 9.23e-3,
                         'lmbda': 8.09e-8, 'rho': 0.9, 'nEpoch': 15, 'nBatch': 100, 'epsilon': 1e-11, 'alpha':0.99}

    '''
    Specify Neural Network parameters in 'NNParams'

    d:                      Input image size 32x32x3
    k:                      Output size (=no. of classes)
    n:                      Number of input images
    m:                      List of sizes of intermediate layers.
    batchNorm:              True/False, if you want to run NN with Batch Normalization
    loadAllBatches:         True/False, whether you want to load all training data (49000 images) 
                            or only one batch (10000 images)
    hiddenLayerSizeList:    specify the number of hidden layers (and with it the no. of neurons in each hidden layer)
                            for eg. [50,30] makes a 3 layer NN with 50 and 30 neurons in 1st and 2nd layer resp.                
    '''
    hiddenLayerSizeList = [50,30] 
    
    NNParams = {'d': 3072, 'k': 10, 'n': 10000, 'm': list(hiddenLayerSizeList), 'batchNorm':True, 'loadAllBatches':False}

    ''' Initialize object '''
    obj = kLayerNN(filePath, GradCheckParams, GradDescentParams, NNParams)

    '''
    Specify mode of program:
                - GradCheck:            True/False. Do you want to compare analytical and numerical gradients?
                - Coarse2FineSearch:    True/False. Do you wanna carry out Coarse to Fine search for eta and 
                                        lambda? Results stored as text files in root folder
                - FinalTest:            True/False. Run gradient descent with chosen parameters (eta, lambda, rho, nEpoch..etc)
    '''
    ProgramMode = {'GradCheck': False, 'Coarse2FineSearch': False, 'FinalTest': True}

    if ProgramMode['GradCheck']:
        ''' 
        Checking Gradients by comparing analytic to numerical gradient
        Change X_DIM and no. of images for training to reduce computations
        '''
        X_DIM = 300     # can change this to 100, 1000 or obj.d (= 3072) etc
        Y_DIM = 100     # this specifies the number of images 
        Temp_Wt = []
        Temp_Wt.append([])
        Temp_Wt.append(obj.W[1][:, 0:X_DIM])
        for i in range(len(obj.m) - 1):
            Temp_Wt.append(obj.W[i+2])
        Temp_Wt.append(obj.W[-1])
        abs_W, abs_b, rel_W, rel_b = obj.CheckGradients(
            obj.Xtrain[0:X_DIM, 0:Y_DIM], obj.Ytrain[0:X_DIM, 0:Y_DIM], Temp_Wt, obj.b, 'slow')

        print('Average absolute errors for Weights and Biases are: ' + str(abs_W) + ' and ' + str(abs_b))
        print('Average relative errors for Weights and Biases are: ' + str(rel_W) + ' and ' + str(rel_b))

    if ProgramMode['Coarse2FineSearch']:
        '''
        Coarse to fine random search for eta and lmbda:
        - Come up with a coarse range for eta and lmbda
        - generate 100 pairs of eta, lmbda and run GD with following settings:
            - 50 epoch, all training and validation data
        - repeat with narrower pair range and 100 epochs 
        - repeat with narrower pair range and 150-200 epochs
        - check accuracy on validation set.
        - save parameter pair and corresponding accuracy in a text file
        '''
        # Test-1
        eta_range = [-3, -1]  # [min, max]
        lmbda_range = [-9, -2]  # [min, max]
        # Since input is required in log scale, hence we take exponents of 10 for range
        # eta_range = [-2.9038, -2.1709]  # [min, max]
        # lmbda_range = [-2.8674, -1.6672]  # [min, max]
        no_of_epochs = 15
        nIter = 75
        obj.CoarseToFineRandomSearch(eta_range, lmbda_range, no_of_epochs, nIter)

    if ProgramMode['FinalTest']:
        obj.FinalTest()


if __name__ == '__main__':
    main()
    plt.show()

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


class kLayerNN(object):
    def __init__(self, filePath, GradCheckParams, GradDescentParams, NNParams):
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

        # Assign all NNParams
        self.d = NNParams['d']
        self.k = NNParams['k']
        self.n = NNParams['n']
        self.m = NNParams['m']
        self.nLayers = len(self.m) + 1

        # Initialize Weights
        self.InitializeWeightAndBias('Gaussian')

    def unpickle(self, file):
        '''
        Function: unpickle
        Input: file name
        Output: data in form of dictionary
        '''
        import pickle
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
        A = np.linalg.inv(np.diag(var + self.epsilon))**0.5
        B = s - mu
        sHat = np.dot(A, B)
        return sHat

    def BatchNormBackPass(self, g, s, mu, var):
        '''
        Input: g (dJ/dsHat), s, mu, var
        Output: g (dJ/ds)
        Comments: Refer to last slide of Lec 4
        '''
        N = g.shape[0]
        Vb = np.diag(var + self.epsilon)
        B = np.reshape(np.diag(s - mu), (-1, 1))
        A = np.dot(g, np.linalg.inv(Vb)**1.5)
        dJdvar = -0.5 * np.sum(np.multiply(A, B), 0)
        E = np.dot(g, np.linalg.inv(Vb)**0.5)
        dJdmu = np.sum(-E, 0)
        X = E
        #Y = np.multiply(dJdvar, np.reshape(np.diag(s - mu), (-1, 1))) * 2/N
        Y = np.dot(np.reshape(np.diag(s-mu), (-1, 1)),
                   np.reshape(dJdvar, (-1, 1)).T) * 2/N
        Z = dJdmu/N
        g1 = X + Y + Z
        return g1

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
            mu.append(np.reshape(np.sum(s[i+1], 1)/N, (-1, 1)))
            # var.append(np.reshape(np.sum(((s[i+1]-mu[i+1])**2), 1)/N, (-1, 1))) #DIAG OF THIS IS SCALAR!!!
            # DIAG OF THIS IS SQUARE MATRIX!!!
            var.append(np.sum(((s[i+1]-mu[i+1])**2), 1)/N)
            sHat.append(self.BatchNormalize(s[i+1], mu[i+1], var[i+1]))
            h.append(np.maximum(0, sHat[i+1]))

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
        # performance1 = (abs(np.array(grad_b[0]) - np.array(numgrad_b[0])))
        # performance2 = (abs(np.array(grad_W[0]) - np.array(numgrad_W[0])))
        # # Layer 2
        # performance3 = (abs(np.array(grad_b[1]) - np.array(numgrad_b[1])))
        # performance4 = (abs(np.array(grad_W[1]) - np.array(numgrad_W[1])))

        '''Method 1: check if relative error is small'''
        # Layer1
        rel_W = []
        rel_b = []
        # for i in range(self.nLayers):
        #     rel_W.append(np.sum(abs(grad_W[i+1] - numgrad_W[i+1])) /
        #                  max(self.eps, np.sum(abs(grad_W[i+1]) + abs(numgrad_W[i+1]))))
        #     rel_b.append(np.sum(abs(grad_b[i+1] - numgrad_b[i+1])) /
        #                  max(self.eps, np.sum(abs(grad_b[i+1]) + abs(numgrad_b[i+1]))))
        for i in range(self.nLayers):
            rel_W.append(np.divide(abs(grad_W[i+1] - numgrad_W[i+1]),
                                   max(self.eps, (abs(grad_W[i+1]) + abs(numgrad_W[i+1])).all())))
            rel_b.append(np.divide(abs(grad_b[i+1] - numgrad_b[i+1]),
                                   max(self.eps, (abs(grad_b[i+1]) + abs(numgrad_b[i+1])).all())))

        # performance5 = np.sum(abs(grad_b[0] - numgrad_b[0])) / \
        #     max(self.eps, np.sum(abs(grad_b[0]) + abs(numgrad_b[0])))
        # performance6 = np.sum(abs(grad_W[0] - numgrad_W[0])) / \
        #     max(self.eps, np.sum(abs(grad_W[0]) + abs(numgrad_W[0])))
        # # Layer2
        # performance5 = np.sum(abs(grad_b[1] - numgrad_b[1])) / \
        #     max(self.eps, np.sum(abs(grad_b[1]) + abs(numgrad_b[1])))
        # performance6 = np.sum(abs(grad_W[1] - numgrad_W[1])) / \
        #     max(self.eps, np.sum(abs(grad_W[1]) + abs(numgrad_W[1])))
        return

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
        # initialize a list of empty lists 'g'
        # g = []
        # for i in range(self.nLayers + 1):
        #     g.append([])
        # gradients for last layer
        g = - (Y - p).T
        grad_b[self.nLayers] = np.sum(g.T, 1)/N
        grad_b[self.nLayers] = np.reshape(grad_b[self.nLayers], (-1, 1))
        grad_W[self.nLayers] = (
            np.dot(g.T, h[self.nLayers-1].T))/N + 2 * self.lmbda * Wt[self.nLayers]
        # Propogate gradient vector gi to previous layer:
        g = np.dot(g, Wt[self.nLayers])
        sTemp = np.where(sHat[self.nLayers-1] > 0, 1, 0)
        g = np.multiply(g, np.reshape(np.diag(sTemp), (-1, 1)))

        for i in range(self.nLayers-1, 0, -1):
            g = self.BatchNormBackPass(g, s[i], mu[i], var[i])
            grad_b[i] = np.sum(g.T, 1)/N
            grad_b[i] = np.reshape(grad_b[i], (-1, 1))
            grad_W[i] = (np.dot(g.T, h[i-1].T))/N + 2 * self.lmbda * Wt[i]
            # Propagate gradient vector to previous layer (if i > 1):
            if i > 1:
                g = np.dot(g, Wt[i])
                sTemp = np.where(sHat[i-1] > 0, 1, 0)
                g = np.multiply(g, np.diag(sTemp))

        return grad_W, grad_b

    def ComputeGradientsNumSlow(self, X, Y, Wt, bias):
        numgrad_W = []
        numgrad_W.append([])
        numgrad_b = []
        numgrad_b.append([])
        numgrad_W.append(np.zeros((list(self.m)[0], Wt[1].shape[1])))
        numgrad_b.append(np.zeros((list(self.m)[0], 1)))

        for i in range(self.nLayers - 2):
            numgrad_W.append(np.zeros((list(self.m)[i], list(self.m)[i+1])))
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
        # if Y.shape[1] == 1:
        #     Y = np.reshape(Y, (-1, 1))
        #     X = np.reshape(X, (-1, 1))

        [grad_W, grad_b] = self.ComputeGradients(X, Y, Wt, bias)

        if mode == 'fast':
            [numgrad_W, numgrad_b] = self.ComputeGradientsNum(X, Y, Wt, bias)
        elif mode == 'slow':
            [numgrad_W, numgrad_b] = self.ComputeGradientsNumSlow(
                X, Y, Wt, bias)

        # Check Performance
        self.PerformanceEval(grad_W, grad_b, numgrad_W, numgrad_b)

    def plotLoss(self, Jtrain, Jval, nEpoch):
        iterations = list(range(1, nEpoch + 1))
        plt.figure()
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
        v_b = []
        v_b.append(np.zeros(bias[1].shape))
        v_b.append(np.zeros(bias[2].shape))
        v_W = []
        v_W.append(np.zeros(Wt[1].shape))
        v_W.append(np.zeros(Wt[2].shape))

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

                for k in range(len(grad_W)):
                    # Momentum Update:
                    v_b[k] = self.rho * v_b[k] + self.eta * grad_b[k]
                    v_W[k] = self.rho * v_W[k] + self.eta * grad_W[k]
                    # Weight/bias update:
                    # HOW IT WAS BEFORE MOMENTUM--
                    # Wstar[k] = Wstar[k] - GDparams.eta * grad_W[k]
                    # bstar[k] = bstar[k] - GDparams.eta * grad_b[k]
                    # HOW IT IS WITH MOMENTUM--
                    Wstar[k] = Wstar[k] - v_W[k]
                    bstar[k] = bstar[k] - v_b[k]

            Jtrain = self.ComputeCost2(Xtrain, Ytrain, Wstar, bstar)
            Jval = self.ComputeCost2(Xval, Yval, Wstar, bstar)
            print('Epoch ' + str(i) + '- Training Error = ' +
                  str(Jtrain) + ', Validation Error = ' + str(Jval))
            JtrainList[i] = Jtrain
            JvalList[i] = Jval
        if PlotLoss == 'true':
            self.plotLoss(JtrainList, JvalList, self.nEpoch)
        return Wstar, bstar

    def CoarseToFineRandomSearch(self, eta_range, lmbda_range, no_of_epochs, nIter):
        # Modify no. of epochs
        self.nEpoch = no_of_epochs
        # Create Text file to store data
        fp = open('parameterTest1.txt', 'w')
        text1 = 'Coarse to Fine parameter search test 1:\neta_range: 1e'+str(eta_range[0]) + '- 1e'+str(eta_range[1])+'\nlmbda range: 1e'+str(lmbda_range[0]) + '- 1e'+str(lmbda_range[1])+'\nnEpochs = ' + \
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
                self.Xtrain, self.Ytrain, self.Xval, self.Yval, self.W, self.b)

            # Compute Accuracy on Validation set
            acc = self.ComputeAccuracy2(self.Xval, self.yval, Wstar, bstar)
            print('>>>>ITERATION ' + str(i) + ':' +
                  'ACCURACY: ' + str(round(acc*100, 2)) + '%<<<<')
            # Write parameters to text file
            fp.write(str(self.eta) + '\t' + str(e) + '\t' +
                     str(self.lmbda) + '\t' + str(f) + '\t' + str(round(acc*100, 2)) + '\n')
        fp.close()
        return

    def FinalTest(self):
        Xtrain_new = np.zeros((3072, 19000))
        Ytrain_new = np.zeros((10, 19000))
        Xval_new = np.zeros((3072, 1000))
        Yval_new = np.zeros((10, 1000))

        Xtrain_new[:, 0:10000] = obj.Xtrain
        Xtrain_new[:, 10000:19000] = obj.Xval[:, 0:9000]
        Ytrain_new[:, 0:10000] = obj.Ytrain
        Ytrain_new[:, 10000:19000] = obj.Yval[:, 0:9000]

        Xval_new = obj.Xval[:, 9000:10000]
        Yval_new = obj.Yval[:, 9000:10000]

        [Wstar, bstar] = obj.MiniBatchGD(
            Xtrain_new, Ytrain_new, Xval_new, Yval_new, obj.W, obj.b, PlotLoss='true')

        # Compute Accuracy
        yval_new = obj.yval[9000:10000]
        acc = obj.ComputeAccuracy2(Xval_new, yval_new, Wstar, bstar)
        print(acc)


def main():
    '''
        - Specify path to Datasets in variable 'filePath'
        - Specify Gradient Checking parameters in 'GradCheckParams'
        - Specify Gradient Descent parameters in 'GradDescentParams'
        - Specify Neural Network parameters in 'NNParams'
    '''
    #filePath = 'C:/Users/Ajinkya/Documents/Python Scripts/Deep Learing in Data Science/'
    #filePath = os.getcwd() + '\\Deep Learning in Data Science'
    filePath = 'C:/Users/Ajinkya/Dropbox/Python Scripts/Deep Learing in Data Science'
    '''
    h:      Step Size
    eps:    epsilon for placing in denominator of relative gradient checking
    tol1:   Tolerance for absolute gradient checking
    '''
    GradCheckParams = {'h': 1e-6, 'eps': 1e-8, 'tol1': 1e-7}

    '''
    sigma:  Variance to initialize random weights
    eta:    Learning Rate
    lmbda:  Regularization Parameter
    rho:    Momentum
    nEpoch: No. of epochs
    nBatch: Batch size
    epsilon:small number used for division in Batch Normalization function
    '''
    GradDescentParams = {'sigma': 0.001, 'eta': 0.01,
                         'lmbda': 0.0, 'rho': 0.9, 'nEpoch': 50, 'nBatch': 100, 'epsilon': 1e-11}

    '''
    d:      Input image size 32x32x3
    k:      Output size (=no. of classes)
    n:      Number of input images
    m:      List of sizes of intermediate layers.
    nLayers:No. of Layers
    '''
    NNParams = {'d': 3072, 'k': 10, 'n': 10000, 'm': list({50, 30})}

    ''' Initialize object '''
    obj = kLayerNN(filePath, GradCheckParams, GradDescentParams, NNParams)

    ''' 
    Checking Gradients by comparing analytic to numerical gradient
    '''
    X_DIM = 1000  # can change this to 100, 1000 or obj.d (= 3072) etc
    Temp_Wt = []
    Temp_Wt.append([])
    Temp_Wt.append(obj.W[1][:, 0:X_DIM])
    for i in range(len(obj.m) - 1):
        Temp_Wt.append(obj.W[i+1])
    Temp_Wt.append(obj.W[-1])
    obj.CheckGradients(
        obj.Xtrain[0:X_DIM, 0:7], obj.Ytrain[0:X_DIM, 0:7], Temp_Wt, obj.b, 'slow')

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
    # # Test-1
    # eta_range = [-3, -2]  # [min, max]
    # lmbda_range = [-3, -1]  # [min, max]
    # # Since input is required in log scale, hence we take exponents of 10 for range
    # # eta_range = [-2.9038, -2.1709]  # [min, max]
    # # lmbda_range = [-2.8674, -1.6672]  # [min, max]
    # no_of_epochs = 15
    # nIter = 75
    # obj.CoarseToFineRandomSearch(eta_range, lmbda_range, no_of_epochs, nIter)

    # obj.FinalTest()

    '''Gradient Descent'''
    [Wstar, bstar] = obj.MiniBatchGD(
        obj.Xtrain[:, 0:10000], obj.Ytrain[:, 0:10000], obj.Xval[:, 0:10000], obj.Yval[:, 0:10000], obj.W, obj.b, PlotLoss='true')

    # Compute Accuracy
    acc = obj.ComputeAccuracy2(obj.Xval, obj.yval, Wstar, bstar)
    print(acc)


if __name__ == '__main__':
    main()
    plt.show()

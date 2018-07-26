'''
Deep Learning for Data Science: Assignment 2
Submitted by: Ajinkya Khoche (khoche@kth.se)
Description -   Add another layer to NN developed in Lab1.
            -   Add momentum
            -   Additional Tests such as initialization, grid search,
            -   Training using Gradient Descent
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

        # Assign all NNParams
        self.d = NNParams['d']
        self.k = NNParams['k']
        self.n = NNParams['n']
        self.m = NNParams['m']
        self.nLayers = NNParams['nLayers']

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
            self.b = []

            self.W.append(np.random.randn(
                list(self.m)[0], self.d) * self.sigma)
            self.b.append(np.zeros((list(self.m)[0], 1)))

            for i in range(len(self.m) - 1):
                self.W.append(np.random.randn(
                    self.m[i+1], self.m[i]) * self.sigma)
                self.b.append(np.zeros((self.m[i+1], 1)))

            self.W.append(np.random.randn(
                self.k, list(self.m)[-1]) * self.sigma)
            self.b.append(np.zeros((self.k, 1)))
        # FUTURE: Add other initializations

    def EvaluateClassifier2(self, x, Wt, bias):
        if x.ndim == 1:
            x = np.reshape(x, (-1, 1))
        s1 = np.dot(Wt[0], x) + bias[0]
        h1 = np.maximum(0, s1)
        s = np.dot(Wt[1], h1) + bias[1]
        # compute softmax function of s
        p = np.exp(s)
        p = p / np.sum(p, axis=0)
        return p, s, h1, s1

    def ComputeCost2(self, X, Y, Wt, bias):
        N = X.shape[1]
        p, _, _, _ = self.EvaluateClassifier2(X, Wt, bias)
        A = np.diag(np.dot(Y.T, p))
        B = -np.log(A)
        C = np.sum(B)
        D = self.lmbda * (np.sum(np.square(Wt[0]))+np.sum(np.square(Wt[1])))

        J = C/N + D
        # J = (np.sum(-np.log(np.dot(Y.T, p))))/M + lmbda * np.sum(np.square(W))
        return J

    def ComputeAccuracy2(self, X, y, Wt, bias):
        N = X.shape[1]
        p, _, _, _ = self.EvaluateClassifier2(X, Wt, bias)
        k = np.argmax(p, axis=0)
        a = np.where((k.T - y) == 0, 1, 0)
        acc = sum(a)/N
        return acc

    def PerformanceEval(self, grad_W, grad_b, numgrad_W, numgrad_b):
        '''Method 2: Check absolute difference'''
        # Layer 1
        performance1 = (abs(np.array(grad_b[0]) - np.array(numgrad_b[0])))
        performance2 = (abs(np.array(grad_W[0]) - np.array(numgrad_W[0])))
        # Layer 2
        performance3 = (abs(np.array(grad_b[1]) - np.array(numgrad_b[1])))
        performance4 = (abs(np.array(grad_W[1]) - np.array(numgrad_W[1])))

        '''Method 1: check if relative error is small'''
        # Layer1
        # performance5 = np.sum(abs(grad_b[0] - numgrad_b[0])) / \
        #     max(self.eps, np.sum(abs(grad_b[0]) + abs(numgrad_b[0])))
        # performance6 = np.sum(abs(grad_W[0] - numgrad_W[0])) / \
        #     max(self.eps, np.sum(abs(grad_W[0]) + abs(numgrad_W[0])))
        # # Layer2
        # performance7 = np.sum(abs(grad_b[1] - numgrad_b[1])) / \
        #     max(self.eps, np.sum(abs(grad_b[1]) + abs(numgrad_b[1])))
        # performance8 = np.sum(abs(grad_W[1] - numgrad_W[1])) / \
        #     max(self.eps, np.sum(abs(grad_W[1]) + abs(numgrad_W[1])))
        performance5 = np.sum(abs(grad_b[0] - numgrad_b[0])) / \
            max(self.eps, np.sum(abs(grad_b[0]) + abs(numgrad_b[0])))
        performance6 = np.sum(abs(grad_W[0] - numgrad_W[0])) / \
            max(self.eps, np.sum(abs(grad_W[0]) + abs(numgrad_W[0])))
        # Layer2
        performance7 = np.sum(abs(grad_b[1] - numgrad_b[1])) / \
            max(self.eps, np.sum(abs(grad_b[1]) + abs(numgrad_b[1])))
        performance8 = np.sum(abs(grad_W[1] - numgrad_W[1])) / \
            max(self.eps, np.sum(abs(grad_W[1]) + abs(numgrad_W[1])))

        return

    def ComputeGradients(self, X, Y, Wt, bias):
        N = X.shape[1]
        grad_W = []
        grad_b = []
        grad_W0 = np.zeros((list(self.m)[0], Wt[0].shape[1]))
        grad_W1 = np.zeros((self.k, list(self.m)[0]))
        grad_b0 = np.zeros((list(self.m)[0], 1))
        grad_b1 = np.zeros((self.k, 1))

        p, s, h1, s1 = self.EvaluateClassifier2(X, Wt, bias)
        # compute g
        g = - (Y - p).T
        grad_b1 = np.sum(g.T, 1)/N
        grad_b1 = np.reshape(grad_b1, (-1, 1))
        grad_W1 = (np.dot(g.T, h1.T))/N + 2 * self.lmbda * Wt[1]

        g1 = np.dot(g, Wt[1])
        #s1_copy = np.copy(s1)
        s1_copy = copy.deepcopy(s1)
        s1_copy[s1_copy > 0] = 1
        s1_copy[s1_copy < 0] = 0
        # g1 = g1 * np.reshape(np.diag(s1_copy), (-1, 1))
        g1 = np.multiply(g1, np.diagonal(np.reshape(s1_copy, (-1, 1))))
        #g1 = np.multiply(g1, np.diag((np.ravel(s1_copy))))

        grad_b0 = np.sum(g1.T, 1)/N
        grad_b0 = np.reshape(grad_b0, (-1, 1))              # grad_b => k x 1
        grad_W0 = (np.dot(g1.T, X.T))/N + 2*self.lmbda * \
            Wt[0]    # grad_W => d x k

        grad_b.append(grad_b0)
        grad_b.append(grad_b1)
        grad_W.append(grad_W0)
        grad_W.append(grad_W1)
        return grad_W, grad_b

    def ComputeGradientsNumSlow(self, X, Y, Wt, bias):
        numgrad_W = []
        numgrad_W.append(np.zeros((list(self.m)[0], Wt[0].shape[1])))
        numgrad_W.append(np.zeros((self.k, list(self.m)[0])))
        numgrad_b = []
        numgrad_b.append(np.zeros((list(self.m)[0], 1)))
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
        Wstar = copy.deepcopy(Wt)
        bstar = copy.deepcopy(bias)

        # Initialize Momentum Vectors
        v_b = []
        v_b.append(np.zeros(bias[0].shape))
        v_b.append(np.zeros(bias[1].shape))
        v_W = []
        v_W.append(np.zeros(Wt[0].shape))
        v_W.append(np.zeros(Wt[1].shape))

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
        if PlotLoss == True:
            self.plotLoss(JtrainList, JvalList, self.nEpoch)
        return Wstar, bstar

    def CoarseToFineRandomSearch(self, eta_range, lmbda_range, no_of_epochs, nIter, filePath):
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

        # Xtrain_new = np.concatenate((X1, X5[:, 0:9000]), axis=1)
        # Ytrain_new = np.concatenate((Y1, Y5[:, 0:9000]), axis=1)
        Xtrain_new = X1
        Ytrain_new = Y1
        Xval_new = X5[:, 9000:10000]
        Yval_new = Y5[:, 9000:10000]
        Xtest_new = X6
        Ytest_new = Y6
        ytest_new = y6

        # Modify no. of epochs
        self.nEpoch = no_of_epochs
        # Create Text file to store data
        fp = open('parameterTest1_5.txt', 'w')
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
                Xtrain_new, Ytrain_new, Xval_new, Yval_new, self.W, self.b, False)

            # Compute Accuracy on Validation set
            acc = self.ComputeAccuracy2(Xtest_new, ytest_new, Wstar, bstar)
            print('>>>>ITERATION ' + str(i) + ':' + ' eta: ' + str(self.eta) + ' lmbda: ' + str(self.lmbda) +
                  ' ACCURACY: ' + str(round(acc*100, 2)) + '%<<<<')
            # Write parameters to text file
            fp.write(str(self.eta) + '\t' + str(e) + '\t' +
                     str(self.lmbda) + '\t' + str(f) + '\t' + str(round(acc*100, 2)) + '\n')
        fp.close()
        return

    def FinalTest(self, filePath):
        # Xtrain_new = np.zeros((3072, 49000))
        # Ytrain_new = np.zeros((10, 49000))
        # Xval_new = np.zeros((3072, 1000))
        # Yval_new = np.zeros((10, 1000))
        # Xtest_new = np.zeros((3072, 10000))
        # Ytest_new = np.zeros((10,10000))

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

        Xtrain_new = np.concatenate((X1, X2, X3, X4, X5[:, 0:9000]), axis=1)
        Ytrain_new = np.concatenate((Y1, Y2, Y3, Y4, Y5[:, 0:9000]), axis=1)
        Xval_new = X5[:, 9000:10000]
        Yval_new = Y5[:, 9000:10000]
        Xtest_new = X6
        Ytest_new = Y6
        ytest_new = y6
        # Xtrain_new[:, 0:10000] = obj.Xtrain
        # Xtrain_new[:, 10000:19000] = obj.Xval[:, 0:9000]
        # Ytrain_new[:, 0:10000] = obj.Ytrain
        # Ytrain_new[:, 10000:19000] = obj.Yval[:, 0:9000]

        # Xval_new = obj.Xval[:, 9000:10000]
        # Yval_new = obj.Yval[:, 9000:10000]

        [Wstar, bstar] = self.MiniBatchGD(
            Xtrain_new, Ytrain_new, Xval_new, Yval_new, self.W, self.b, PlotLoss=True)

        # Compute Accuracy
        #yval_new = obj.yval[9000:10000]
        acc = obj.ComputeAccuracy2(Xtest_new, ytest_new, Wstar, bstar)
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
    GradCheckParams = {'h': 1e-6, 'eps': 1e-3, 'tol1': 1e-7}

    '''
    sigma:  Variance to initialize random weights
    eta:    Learning Rate
    lmbda:  Regularization Parameter
    rho:    Momentum
    nEpoch: No. of epochs
    nBatch: Batch size
    '''
    GradDescentParams = {'sigma': 0.001, 'eta': 5.58e-3,
                         'lmbda': 6.69e-8, 'rho': 0.9, 'nEpoch': 20, 'nBatch': 100}

    '''
    d:      Input image size 32x32x3
    k:      Output size (=no. of classes)
    n:      Number of input images
    m:      List of sizes of intermediate layers.
    nLayers:No. of Layers
    '''
    NNParams = {'d': 3072, 'k': 10, 'n': 10000, 'm': {50}, 'nLayers': len('m')}

    ''' Initialize object '''
    obj = kLayerNN(filePath, GradCheckParams, GradDescentParams, NNParams)

    ''' 
    Checking Gradients by comparing analytic to numerical gradient
    '''
    # X_DIM = 1000  # can change this to 100, 1000 etc
    # Temp_Wt = []
    # Temp_Wt.append(obj.W[0][:, 0:X_DIM])
    # Temp_Wt.append(obj.W[1])
    # obj.CheckGradients(
    #     obj.Xtrain[0:X_DIM, 0:1], obj.Ytrain[0:X_DIM, 0:1], Temp_Wt, obj.b, 'slow')

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
    lmbda_range = [-9, -3]  # [min, max]
    # Since input is required in log scale, hence we take exponents of 10 for range
    # eta_range = [-2.9038, -2.1709]  # [min, max]
    # lmbda_range = [-2.8674, -1.6672]  # [min, max]
    no_of_epochs = 15
    nIter = 75
    obj.CoarseToFineRandomSearch(
        eta_range, lmbda_range, no_of_epochs, nIter, filePath)

    # obj.FinalTest(filePath)

    '''Gradient Descent'''
    # [Wstar, bstar] = obj.MiniBatchGD(
    #     obj.Xtrain[:, 0:10000], obj.Ytrain[:, 0:10000], obj.Xval[:, 0:10000], obj.Yval[:, 0:10000], obj.W, obj.b, PlotLoss=True)

    # Compute Accuracy
    #acc = obj.ComputeAccuracy2(obj.Xval, obj.yval, Wstar, bstar)
    # print(acc)


if __name__ == '__main__':
    main()
    plt.show()

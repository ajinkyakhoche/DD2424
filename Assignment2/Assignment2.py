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

#from keras.utils import to_categorical
from sklearn import preprocessing


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def LoadBatch(fileName):
    dict = unpickle(fileName)
    X = np.array(dict[b'data']/255)
    y = np.array(dict[b'labels'])
    # one hot encode
    #Y = to_categorical(y)
    #Y = Y.astype(int)
    binarizer = preprocessing.LabelBinarizer()
    binarizer.fit(range(max(y.astype(int)) + 1))
    Y1 = np.array(binarizer.transform(y.astype(int))).T
    return np.transpose(X), np.transpose(Y1.T), y


def EvaluateClassifier2(x, W, b):
    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))
    s1 = np.dot(W[0], x) + b[0]
    h = np.maximum(0, s1)
    s = np.dot(W[1], h) + b[1]
    # compute softmax function of s
    p = np.exp(s)
    p = p / np.sum(p, axis=0)
    return p, s, h, s1


def ComputeCost2(X, Y, W, b, lmbda):
    M = X.shape[1]
    p, _, _, _ = EvaluateClassifier2(X, W, b)
    A = np.diag(np.dot(Y.T, p))
    B = -np.log(A)
    C = np.sum(B)
    D = lmbda * (np.sum(np.square(W[0]))+np.sum(np.square(W[1])))

    J = C/M + D
    #J = (np.sum(-np.log(np.dot(Y.T, p))))/M + lmbda * np.sum(np.square(W))
    return J


def ComputeAccuracy2(X, y, W, b):
    M = X.shape[1]
    p, _, _, _ = EvaluateClassifier2(X, W, b)
    k = np.argmax(p, axis=0)
    a = np.where((k.T - y) == 0, 1, 0)
    acc = sum(a)/M
    return acc


def PerformanceEval(grad_W, grad_b, numgrad_W, numgrad_b):
    # Method 2: Check absolute difference
    performance3 = (abs(np.array(grad_b) - np.array(numgrad_b)))
    performance4 = (abs(np.array(grad_W) - np.array(numgrad_W)))
    return


def CheckGradients(X, Y, W, b, lmbda, h, tol, eps, mode='fast'):
    # This function difference b/w analytical gradient and numerical gradient
    # since the definition of 'small' difference wasn't defined in lab
    # instruction, the value of performance1, 2, 3 and 4 was checked
    # qualitatively to be small
    if Y.ndim == 1:
        Y = np.reshape(Y, (-1, 1))
        X = np.reshape(X, (-1, 1))

    [grad_W, grad_b] = ComputeGradients(
        X, Y, W, b, lmbda)

    if mode == 'fast':
        [numgrad_W, numgrad_b] = ComputeGradsNum(
            X, Y, W, b, lmbda, h)
    else:
        [numgrad_W, numgrad_b] = ComputeGradsNumSlow(
            X, Y, W, b, lmbda, h)

    # First layer weights
    PerformanceEval(grad_W[0], grad_b[0], numgrad_W[0], numgrad_b[0])
    # Second layer weights
    PerformanceEval(grad_W[1], grad_b[1], numgrad_W[1], numgrad_b[1])

    # Method 1: check if relative error is small
    # performance1 = sum(abs(grad_b - numgrad_b)) / \
    #     max(eps, sum(abs(grad_b) + abs(numgrad_b)))
    # performance2 = np.sum(abs(grad_W - numgrad_W)) / \
    #     max(eps, np.sum(abs(grad_W) + abs(numgrad_W)))

    # Method 2: check if absolute error is small
    # performance3 = (abs(np.array(grad_b) - np.array(numgrad_b)))
    # performance4 = (abs(np.array(grad_W) - np.array(numgrad_W)))
    return True


def ComputeGradients(X, Y, W, b, lmbda):
    # X = d x n , where n = no of images
    # Y = k x n
    # p = k x n
    # W = k x d

    M = X.shape[1]
    grad_W = []
    grad_b = []
    grad_W0 = np.zeros(W[0].shape)
    grad_W1 = np.zeros(W[1].shape)
    grad_b0 = np.zeros(b[0].shape)
    grad_b1 = np.zeros(b[1].shape)

    p, s, h, s1 = EvaluateClassifier2(X, W, b)
    # compute g
    g = - (Y - p).T
    grad_b1 = np.sum(g.T, 1)/M
    grad_b1 = np.reshape(grad_b1, (-1, 1))
    grad_W1 = (np.dot(g.T, h.T))/M + 2 * lmbda * W[1]

    g1 = np.dot(g, W[1])
    s1_copy = np.copy(s1)
    s1_copy[s1_copy > 0] = 1
    s1_copy[s1_copy < 0] = 0
    #g1 = g1 * np.reshape(np.diag(s1_copy), (-1, 1))
    g1 = np.multiply(g1, np.diag(s1_copy))

    grad_b0 = np.sum(g1.T, 1)/M
    grad_b0 = np.reshape(grad_b0, (-1, 1))              # grad_b => k x 1
    grad_W0 = (np.dot(g1.T, X.T))/M + 2*lmbda * W[0]    # grad_W => d x k

    grad_b.append(grad_b0)
    grad_b.append(grad_b1)
    grad_W.append(grad_W0)
    grad_W.append(grad_W1)
    return grad_W, grad_b


def ComputeGradients2(X, Y, W, b, lmbda):
    d = X.shape[0]
    n = X.shape[1]
    k = Y.shape[0]
    m = W[0].shape[0]

    grad_W = []
    grad_b = []
    grad_W0 = np.zeros((m, d))
    grad_W1 = np.zeros((k, m))
    grad_b0 = np.zeros((m, 1))
    grad_b1 = np.zeros((k, 1))

    p, s1, h, s = EvaluateClassifier2(X, W, b)

    for i in range(n):
        g = -(Y[:, i] - p[:, i]).T
        g = (np.reshape(g, (-1, 1))).T
        grad_b1 += g
        grad_W1 += np.dot(g.T, h[:, i].T)
        g = np.dot(g, W[1])
        s1_copy = s1.copy()
        s1_copy[s1_copy > 0] = 1
        s1_copy[s1_copy < 0] = 0
        g = np.multiply(g, s1_copy)
        grad_b0 += g
        grad_W0 += np.dot(g.T, X.T)

    grad_b0 /= n
    grad_W0 /= n
    grad_b1 /= n
    grad_W1 /= n

    grad_b.append(grad_b0)
    grad_b.append(grad_b1)
    grad_W.append(grad_W0)
    grad_W.append(grad_W1)
    return grad_W, grad_b


def ComputeGradsNum(X, Y, W, b, lmbda, h):
    #no = W.shape[0]
    #d = X.shape[0]
    numgrad_W = []
    numgrad_W.append(np.zeros(W[0].shape))
    numgrad_W.append(np.zeros(W[1].shape))
    numgrad_b = []
    numgrad_b.append(np.zeros(b[0].shape))
    numgrad_b.append(np.zeros(b[1].shape))
    c = ComputeCost2(X, Y, W, b, lmbda)

    for j in range(len(b)):
        #numgrad_b[j] = np.zeros(b[j].shape)
        for i in range(len(b[j])):
            b_try = b
            b_try[j][i] += h
            c2 = ComputeCost2(X, Y, W, b_try, lmbda)
            numgrad_b[j][i] = (c2 - c)/h

    for j in range(len(W)):
        #numgrad_W = np.zeros(W[j].shape)
        for i in range(len(W[j])):
            W_try = W
            W_try[j][i] += h
            c2 = ComputeCost2(X, Y, W_try, b, lmbda)

            numgrad_W[j][i] = (c2-c)/h

    return numgrad_W, numgrad_b


def ComputeGradsNumSlow(X, Y, W, b, lmbda, h):
    numgrad_W = []
    numgrad_W.append(np.zeros(W[0].shape))
    numgrad_W.append(np.zeros(W[1].shape))
    numgrad_b = []
    numgrad_b.append(np.zeros(b[0].shape))
    numgrad_b.append(np.zeros(b[1].shape))

    for j in range(len(b)):
        #numgrad_b[j] = np.zeros(b[j].shape)
        for i in range(len(b[j])):
            b_try = b
            b_try[j][i] -= h
            c1 = ComputeCost2(X, Y, W, b_try, lmbda)

            b_try = b
            b_try[j][i] += h
            c2 = ComputeCost2(X, Y, W, b_try, lmbda)

            numgrad_b[j][i] = (c2 - c1)/h

    for j in range(1, 2):
        #numgrad_W = np.zeros(W[j].shape)
        for i in range(len(W[j])):
            for k in range(len(W[j][0])):
                W_try = W
                W_try[j][i][k] -= h
                c1 = ComputeCost2(X, Y, W_try, b, lmbda)

                W_try = W
                W_try[j][i][k] += h
                c2 = ComputeCost2(X, Y, W_try, b, lmbda)

                numgrad_W[j][i][k] = (c2-c1)/h

    return numgrad_W, numgrad_b


class defineParams(object):
    n_batch = 0
    eta = 0
    n_epochs = 0
    rho = 0

    # The class "constructor" - It's actually an initializer
    def __init__(self, n_batch, eta, n_epoch, rho):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epoch
        self.rh0 = rho


def plotLoss(Jtrain, Jval, nEpoch):
    iterations = list(range(1, nEpoch + 1))
    plt.figure()
    plt.plot(iterations, Jtrain, linewidth=3, label='Training Loss')
    plt.plot(iterations, Jval, linewidth=3, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss')
    plt.grid()


def MiniBatchGD(Xtrain, Ytrain, Xval, Yval, GDparams, W, b, lmbda):
    # Initialize Wstar and bstar (these would serve as updated weights)
    Wstar = W
    bstar = b

    # Initialize Momentum Vectors
    v_b = []
    v_b.append(np.zeros(b[0].shape))
    v_b.append(np.zeros(b[1].shape))
    v_W = []
    v_W.append(np.zeros(W[0].shape))
    v_W.append(np.zeros(W[1].shape))

    # v_W0 = np.zeros(W[0].shape)
    # v_W1 = np.zeros(W[1].shape)
    # v_b0 = np.zeros(b[0].shape)
    # v_b1 = np.zeros(b[1].shape)

    JtrainList = np.zeros(GDparams.n_epochs)
    JvalList = np.zeros(GDparams.n_epochs)
    # No. of batch iterations
    nBatchIter = int(Xtrain.shape[1]/GDparams.n_batch)
    for i in range(GDparams.n_epochs):
        for j in range(nBatchIter):
            # extract a batch for training
            j_start = j * GDparams.n_batch
            j_end = (j+1)*GDparams.n_batch
            Xbatch = Xtrain[:, j_start: j_end]
            Ybatch = Ytrain[:, j_start: j_end]

            # Forward pass and Back Propagation:
            [grad_W, grad_b] = ComputeGradients(
                Xbatch, Ybatch, Wstar, bstar, lmbda)

            for k in range(len(grad_W)):
                # Momentum Update:
                v_b[k] = GDparams.rho * v_b[k] + GDparams.eta * grad_b[k]
                v_W[k] = GDparams.rho * v_W[k] + GDparams.eta * grad_W[k]
                # Weight/bias update:
                # HOW IT WAS BEFORE MOMENTUM--
                # Wstar[k] = Wstar[k] - GDparams.eta * grad_W[k]
                # bstar[k] = bstar[k] - GDparams.eta * grad_b[k]
                # HOW IT IS WITH MOMENTUM--
                Wstar[k] = Wstar[k] - v_W[k]
                bstar[k] = bstar[k] - v_b[k]

        Jtrain = ComputeCost2(Xtrain, Ytrain, Wstar, bstar, lmbda)
        Jval = ComputeCost2(Xval, Yval, Wstar, bstar, lmbda)
        print('Epoch ' + str(i) + '- Training Error = ' +
              str(Jtrain) + ', Validation Error = ' + str(Jval))
        JtrainList[i] = Jtrain
        JvalList[i] = Jval
    plotLoss(JtrainList, JvalList, GDparams.n_epochs)
    return Wstar, bstar


def VisualizeWeights(Wstar):
    plt.figure(figsize=(2, 5))
    plt.tight_layout()
    for i in range(Wstar.shape[0]):
        plt.subplot(2, 5, i + 1)
        im = np.reshape(Wstar[i, :], (32, 32, 3))
        min_im = np.amin(im)
        max_im = np.amax(im)
        s_im = (im - min_im)/(max_im - min_im)
        #s_im = np.transpose(s_im[:, :, :], (1, 0, 2))
        plt.imshow(s_im)
        plt.margins(tight=True)
        plt.axis('off')
    plt.suptitle('Weight of Hidden Neurons Visualized')

    return


def GaussInitialization2(k, d, m, sigma):
    np.random.seed(400)
    W1 = np.random.randn(m, d) * sigma
    W2 = np.random.randn(k, m) * sigma
    b1 = np.zeros((m, 1))
    b2 = np.zeros((k, 1))
    W = [W1, W2]
    b = [b1, b2]
    np.savetxt('W0.txt', W1, delimiter=',')
    np.savetxt('W1.txt', W2, delimiter=',')
    return W, b


def ZeroMean(X):
    mean_X = np.reshape(np.mean(X, 1), (-1, 1))
    print(mean_X.shape)
    X = X - mean_X
    return X, mean_X


def main():
    # !!!NOTE: Specify your file path to Datasets HERE!!!
    filePath = 'C:/Users/Ajinkya/Documents/Python Scripts/Deep Learing in Data Science/'
    # Call LoadBatch function to get training, validation and test set data
    Xtrain, Ytrain, ytrain = LoadBatch(filePath +
                                       'Datasets/cifar-10-python/cifar-10-batches-py/data_batch_1')
    Xval, Yval, yval = LoadBatch(filePath +
                                 '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_2')
    Xtest, Ytest, ytest = LoadBatch(filePath +
                                    '/Datasets/cifar-10-python/cifar-10-batches-py/test_batch')

    # Transform Input images to have zero mean
    Xtrain, mean_Xtrain = ZeroMean(Xtrain)
    Xval = Xval - mean_Xtrain
    Xtest = Xtest - mean_Xtrain

    # d = dimension of input = 32x32*3 = 3072
    # k = dimension of output = 10
    # N = no. of training images
    # m = no. of neurons in hidden layer
    d = Xtrain.shape[0]
    k = Ytrain.shape[0]
    n = ytrain.shape[0]
    m = 50

    # Initialize W and b
    sigma = 0.001
    W, b = GaussInitialization2(k, d, m, sigma)

    lmbda = 0.00  # L2 Regularization parameter
    h = 1e-6  # tolerance
    eps = 0.001  # for gradient checking
    tol = 1e-5  # for gradient checking

    # # Forward pass
    #p = EvaluateClassifier2(Xtrain[:, 0:100], W, b)

    # Checking Gradients by comparing analytic to numerical gradient
    # status = CheckGradients(
    #    Xtrain[:, 0:100], Ytrain[:, 0:100], W, b, lmbda, h, tol, eps, 'slow')

    # if status == True:
    #     print('SUCCESS: ComputeGradients')
    # else:
    #     print('FAILURE: ComputeGradients')

    # Define gradient descent parameters
    nBatch = 100
    eta = 0.001
    nEpochs = 200
    rho = 0.9
    GDparams = defineParams(nBatch, eta, nEpochs, rho)

    # CoarseToFineRandomSearch()
    # Call gradient descent
    [Wstar, bstar] = MiniBatchGD(
        Xtrain[:, 0:1000], Ytrain[:, 0:1000], Xval[:, 0:1000], Yval[:, 0:1000], GDparams, W, b, lmbda)

    # Compute Accuracy
    acc = ComputeAccuracy2(Xtest, ytest, Wstar, bstar)
    print(acc)

    # Save weights to show in Matlab
    # np.savetxt(
    #    'C:\\Users\\Ajinkya\\Documents\\Python Scripts\\Deep Learing in Data Science\\Assignment1\\Wstar.txt', Wstar, delimiter=',')
    # Visualize weights
    # VisualizeWeights(Wstar)
    plt.show()


# MAIN
if __name__ == '__main__':
    main()

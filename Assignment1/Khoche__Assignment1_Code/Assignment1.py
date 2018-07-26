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


def EvaluateClassifier(x, W, b):
    if x.ndim == 1:
        x = np.reshape(x, (-1, 1))
    s = np.dot(W, x) + b
    # compute softmax function of s
    p = np.exp(s)
    p = p / np.sum(p, axis=0)
    return p


def ComputeCost(X, Y, W, b, lmbda):
    M = X.shape[1]
    p = EvaluateClassifier(X, W, b)
    A = np.diag(np.dot(Y.T, p))
    B = -np.log(A)
    C = np.sum(B)
    D = lmbda * np.sum(np.square(W))

    J = C/M + D
    #J = (np.sum(-np.log(np.dot(Y.T, p))))/M + lmbda * np.sum(np.square(W))
    return J


def ComputeAccuracy(X, y, W, b):
    M = X.shape[1]
    p = EvaluateClassifier(X, W, b)
    k = np.argmax(p, axis=0)
    a = np.where((k.T - y) == 0, 1, 0)
    acc = sum(a)/M
    return acc


def CheckGradients(X, Y, p, W, b, lmbda, h, tol, eps, mode='fast'):
    # This function difference b/w analytical gradient and numerical gradient
    # since the definition of 'small' difference wasn't defined in lab
    # instruction, the value of performance1, 2, 3 and 4 was checked
    # qualitatively to be small
    if Y.ndim == 1:
        Y = np.reshape(Y, (-1, 1))
        X = np.reshape(X, (-1, 1))

    [grad_W1, grad_b1] = ComputeGradients(X, Y, p, W, lmbda)

    if mode == 'fast':
        [grad_b2, grad_W2] = ComputeGradsNum(X, Y, W, b, lmbda, h)
    else:
        [grad_b2, grad_W2] = ComputeGradsNumSlow(X, Y, W, b, lmbda, h)

    # Method 1: check if relative error is small
    performance1 = np.sum(abs(grad_b1 - grad_b2)) / \
        max(eps, np.sum(abs(grad_b1) + abs(grad_b2)))
    performance2 = np.sum(abs(grad_W1 - grad_W2)) / \
        max(eps, np.sum(abs(grad_W1) + abs(grad_W2)))

    # Method 2: check if absolute error is small
    performance3 = (abs(grad_b1 - grad_b2))
    performance4 = (abs(grad_W1 - grad_W2))

    if performance1 < tol and performance2 < tol and np.all(performance3 < tol) and np.all(performance4 < tol):
        status = True
    else:
        status = False

    if status == True:
        print('SUCCESS: ComputeGradients')
    else:
        print('FAILURE: ComputeGradients')
    return


def ComputeGradients(X, Y, p, W, lmbda):
    # X = d x n , where n = no of images
    # Y = k x n
    # p = k x n
    # W = k x d
    # Refer to slide 81 of Lecture 3 on Backpropagation
    M = X.shape[1]
    g = - (Y - p).T
    # if Y.ndim == 1:
    #     g = - Y.T / (np.dot(Y.T, p))
    # else:
    #     g = - (np.dot(np.linalg.inv(np.dot(Y.T, p)), Y.T))  # dJ/dp => n x k
    # g = np.dot(g, (np.diagonal(p) - np.dot(p, p.T)))  # dJ/ds => n x k
    # g = np.dot(g, np.eye(Y.shape[0]))  # dJ/dz => n x k
    grad_b = np.mean(g.T, 1)
    grad_b = np.reshape(grad_b, (-1, 1))  # grad_b => k x 1
    grad_W = (np.dot(g.T, X.T))/M + 2*lmbda * W  # grad_W => d x k
    return grad_W, grad_b


def ComputeGradsNum(X, Y, W, b, lmbda, h):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros_like(W)
    grad_b = np.zeros((no, 1))

    c = ComputeCost(X, Y, W, b, lmbda)

    for i in range(b.shape[0]):
        b_try = b
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lmbda)
        grad_b[i] = (c2 - c)/h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i][j] += h
            c2 = ComputeCost(X, Y, W_try, b, lmbda)

            grad_W[i][j] = (c2-c)/h

    return grad_b, grad_W


def ComputeGradsNumSlow(X, Y, W, b, lmbda, h):
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros_like(W)
    grad_b = np.zeros((no, 1))

    for i in range(b.shape[0]):
        b_try = b
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lmbda)

        b_try = b
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lmbda)
        grad_b[i] = (c2 - c1)/(1*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W
            W_try[i][j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lmbda)

            W_try = W
            W_try[i][j] += h
            c2 = ComputeCost(X, Y, W_try, b, lmbda)

            grad_W[i][j] = (c2-c1)/(1*h)

    return grad_b, grad_W


class defineParams(object):
    n_batch = 0
    eta = 0
    n_epochs = 0

    # The class "constructor" - It's actually an initializer
    def __init__(self, n_batch, eta, n_epoch):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epoch


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
    Wstar = W
    bstar = b
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

            # Forward pass
            p = EvaluateClassifier(Xbatch, Wstar, bstar)
            # Back Propagation
            [grad_W, grad_b] = ComputeGradients(
                Xbatch, Ybatch, p, Wstar, lmbda)
            # Weight/bias update
            Wstar = Wstar - GDparams.eta * grad_W
            bstar = bstar - GDparams.eta * grad_b
        Jtrain = ComputeCost(Xtrain, Ytrain, Wstar, bstar, lmbda)
        Jval = ComputeCost(Xval, Yval, Wstar, bstar, lmbda)
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


def GaussInitialization(k, d, sigma):
    # np.random.seed(400)
    W = np.random.randn(k, d) * sigma
    b = np.random.randn(k, 1) * sigma
    return W, b


def main():
    # !!!NOTE: Specify your file path to Datasets HERE!!!
    filePath = 'C:/Users/Ajinkya/Documents/Python Scripts/Deep Learing in Data Science/'
    # Call LoadBatch function to get training, validation and test set data
    Xtrain, Ytrain, ytrain = LoadBatch(filePath +
                                       'Datasets/cifar-10-python/cifar-10-batches-py/data_batch_1')
    # print(Xtrain.shape)
    # print(Ytrain.shape)
    # print(ytrain.shape)
    Xval, Yval, yval = LoadBatch(filePath +
                                 '/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_2')
    Xtest, Ytest, ytest = LoadBatch(filePath +
                                    '/Datasets/cifar-10-python/cifar-10-batches-py/test_batch')

    # d = dimension of input = 32x32*3 = 3072
    # k = dimension of output = 10
    # N = no. of training images
    d = Xtrain.shape[0]
    k = Ytrain.shape[0]
    N = ytrain.shape[0]

    # Initialize W and b
    sigma = 0.01
    W, b = GaussInitialization(k, d, sigma)

    lmbda = 1   # L2 Regularization parameter
    h = 1e-6    # tolerance
    eps = 0.001  # for gradient checking
    tol = 1e-3  # Tolerance for gradient checking

    # # Forward pass
    p = EvaluateClassifier(Xtrain[:, 0:1], W[:, :], b)

    # Checking Gradients by comparing analytic to numerical gradient
    # CheckGradients(
    #    Xtrain[:, 0:1], Ytrain[:, 0:1], p, W[:, :], b, lmbda, h, tol, eps, 'slow')

    # Define gradient descent parameters
    # nBatch, eta, nEpochs
    GDparams = defineParams(100, 0.01, 40)

    # Call gradient descent
    [Wstar, bstar] = MiniBatchGD(
        Xtrain, Ytrain, Xval, Yval, GDparams, W, b, lmbda)

    # Compute Accuracy
    acc = ComputeAccuracy(Xtest, ytest, Wstar, bstar)
    print(acc)

    # Save weights to show in Matlab
    np.savetxt(
        'C:\\Users\\Ajinkya\\Documents\\Python Scripts\\Deep Learing in Data Science\\Assignment1\\Wstar.txt', Wstar, delimiter=',')
    # Visualize weights
    VisualizeWeights(Wstar)
    plt.show()


# MAIN
if __name__ == '__main__':
    main()

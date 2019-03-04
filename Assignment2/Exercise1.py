import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def pre_train(is_visualized=False, N=100, D=2, K=3):
    """

    used to generate the data to construct the neural network mode
    :param is_visualized:
    :param N: number of points per class
    :param D: dimensionality
    :param K: number of classes
    :return: x1: argument input vectors
    :return: y1
    """

    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        # radius
        r = np.linspace(0.0, 1, N)
        # theta
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # argument input vectors
    x1 = Variable(
        torch.from_numpy(X).type(torch.FloatTensor),
        requires_grad=False
    )
    el1 = torch.cat((torch.ones(100), torch.zeros(200)), 0)
    el2 = torch.cat((torch.zeros(100), torch.ones(100), torch.zeros(100)), 0)
    el3 = torch.cat((torch.zeros(200), torch.ones(100)), 0)
    y1 = Variable(torch.stack((el1, el2, el3), 1), requires_grad=False)

    if is_visualized:
        # visualize the data:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    return x1, y1


def gradient_descent_with_relu(
        x1, y1, learning_rate=0.0001, n_units=50, n_iterations=100_000,
        N=100, D=3, K=3
):
    """
    gradient descent for the neural network model with ReLU
    :param x1: the input vector
    :param y1:
    :param learning_rate: the learning rate
    :param n_units: the number of units in the hidden layer
    :param n_iterations: the number of hte iterations for gradient descent
    :param N: number of points per class
    :param D: dimensionality
    :param K: number of classes
    :return: a numpy array of loss values over each gradient descent
    """
    # define loss value record
    loss_values = np.zeros(n_iterations)

    # parameters
    w1 = Variable(0.5 * torch.randn(D, n_units), requires_grad=True)
    b1 = Variable(torch.randn(1, n_units), requires_grad=True)
    w2 = Variable(0.5 * torch.randn(n_units, K), requires_grad=True)
    b2 = Variable(torch.randn((1, K)), requires_grad=True)

    # run with different learning rates
    # gradient descent loop
    for i in range(n_iterations):

        # Forward pass
        hout = x1.mm(w1) + b1
        h_relu = hout.clamp(min=0)
        output = h_relu.mm(w2) + b2  # output of NN
        # scores = output.clamp(min=0)
        scores = output

        # compute the loss
        loss = (scores - y1).pow(2).sum()
        loss_values[i] = loss

        loss.backward()

        if i % 10_000 == 0:
            print(i, loss)

        # perform a parameter update
        w2.data -= learning_rate * w2.grad.data
        w1.data -= learning_rate * w1.grad.data
        b2.data -= learning_rate * b2.grad.data
        b1.data -= learning_rate * b1.grad.data

        # Manually zero the gradient after the backward pass
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        b1.grad.data.zero_()

    # evaluate training set accuracy
    desired_class = torch.cat(
        (torch.zeros(100), torch.ones(100), torch.add(torch.ones(100), 1)),
        0
    )
    hout = x1.mm(w1) + b1
    h_relu = hout.clamp(min=0)
    output = h_relu.mm(w2) + b2
    # scores = output.clamp(min=0)
    scores = output
    predicted_class = torch.max(scores, 1)
    predictedClass = np.asarray(predicted_class[1])
    predictedTC = torch.FloatTensor(predictedClass)
    # print(predicted_class[1])
    predicted = torch.squeeze(predictedTC)
    err = torch.eq(desired_class, predicted)
    error = N * K - torch.sum(err)
    print('The number of misclassified examples: ', error)

    accuracy = torch.sum(err).numpy() / (N * K)
    return loss_values, accuracy


def gradient_descent_with_sigmoid_function(
        x1, y1, learning_rate=0.0001, n_units=50, n_iterations=100_000,
        N=100, D=2, K=3
):
    """
    gradient descent for the neural network model with sigmoid function
    :param x1: the input vector
    :param y1:
    :param learning_rate: the learning rate
    :param n_units: the number of units in the hidden layer
    :param n_iterations: the number of hte iterations for gradient descent
    :param N: number of points per class
    :param D: dimensionality
    :param K: number of classes
    :return: a numpy array of loss values over each gradient descent
    """
    # define loss value record
    loss_values = np.zeros(n_iterations)

    # parameters
    w1 = Variable(0.5 * torch.randn(D, n_units), requires_grad=True)
    b1 = Variable(torch.randn(1, n_units), requires_grad=True)
    w2 = Variable(0.5 * torch.randn(n_units, K), requires_grad=True)
    b2 = Variable(torch.randn((1, K)), requires_grad=True)

    # run with different learning rates
    # gradient descent loop
    for i in range(n_iterations):

        # Forward pass
        hout = x1.mm(w1) + b1
        h_sigmoidal = 1.0 / (1.0 + torch.exp(-hout))
        # output of the NN
        output = h_sigmoidal.mm(w2) + b2
        scores = output

        # compute the loss
        loss = (scores - y1).pow(2).sum()
        loss_values[i] = loss

        loss.backward()

        if i % 10_000 == 0:
            print(i, loss)

        # perform a parameter update
        w2.data -= learning_rate * w2.grad.data
        w1.data -= learning_rate * w1.grad.data
        b2.data -= learning_rate * b2.grad.data
        b1.data -= learning_rate * b1.grad.data

        # Manually zero the gradient after the backward pass
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        b1.grad.data.zero_()

    # evaluate training set accuracy
    desired_class = torch.cat(
        (torch.zeros(100), torch.ones(100), torch.add(torch.ones(100), 1)),
        0
    )
    hout = x1.mm(w1) + b1
    h_relu = hout.clamp(min=0)
    output = h_relu.mm(w2) + b2
    # scores = output.clamp(min=0)
    scores = output
    predicted_class = torch.max(scores, 1)
    predictedClass = np.asarray(predicted_class[1])
    predictedTC = torch.FloatTensor(predictedClass)
    # print(predicted_class[1])
    predicted = torch.squeeze(predictedTC)
    err = torch.eq(desired_class, predicted)
    error = N * K - torch.sum(err)
    print('The number of misclassified examples: ', error)

    accuracy = torch.sum(err).numpy() / (N * K)
    return loss_values, accuracy


def loss_values_over_iterations_with_different_learning_rates(
        x1,
        y1,
        N=100,
        D=2,
        K=3,
        learning_rates=[0.0001],
        gd_function=0
):
    """
    loss values over iterations with different learning rates
    :param x1: input
    :param y1: class vector
    :param N: number of points per class
    :param D: dimensionality
    :param K: number of classes
    :param learning_rates: a list of learning rates to be tested
    :param gd_function: {0: gradient descent with ReLU,
                else: gradient descent with sigmoid function}

    :return:
    """
    if gd_function == 0:
        gradient_descent = gradient_descent_with_relu
        title = 'Loss values over the iteration ' \
                'with different learning rates using ReLU'
    else:
        gradient_descent = gradient_descent_with_sigmoid_function
        title = 'Loss values over the iteration ' \
                'with different learning rates using sigmoid function'

    n_rates = len(learning_rates)
    n_iterations = 20_000
    loss_array = np.empty((n_rates, n_iterations))

    for i in range(n_rates):
        loss_array[i], _ = gradient_descent(
            x1=x1,
            y1=y1,
            learning_rate=learning_rates[i],
            n_iterations=n_iterations,
            N=N,
            D=D,
            K=K,

        )

    # print the first 10 loss values
    n_first_rows = 10
    for i in range(n_first_rows):
        for j in range(n_rates):
            print('{0: 10.3f}\t'.format(loss_array[j, i]), end='')
        print("")

    # plot the loss values
    figure3_1 = plt.figure()
    plot_x = [n for n in range(1, n_iterations + 1)]
    for i in range(n_rates):
        plt.plot(plot_x, loss_array[i, :])

    plt.title(title)
    plt.xlabel('iteration number')
    plt.ylabel('loss value')
    plt.legend(
        ['learning rate = {}'.format(learning_rate)
         for learning_rate in learning_rates],
        loc='upper right'
    )
    plt.ylim(0, 500.0)
    plt.show()


def exercise1_1():
    # generate hyper-parameters ranging from 0.1 to 10^-9
    rate_list = [math.pow(10, -n) for n in range(1, 1 + 9)]
    x1, y1 = pre_train()
    loss_values_over_iterations_with_different_learning_rates(
        x1=x1,
        y1=y1,
        N=100,
        D=2,
        K=3,
        learning_rates=rate_list,
        gd_function=0
    )


def accuracy_over_different_sizes_of_hidden_layer(
        x1,
        y1,
        N=100,
        D=2,
        K=3,
        n_units=[20, 30, 40, 50],
        gd_function=0
):
    """
    exercise1-3-2
    :param x1: input
    :param y1: class vector
    :param N: number of points per class
    :param D: dimensionality
    :param K: number of classes
    :param n_units: list of sizes of the hidden layers to be tested.
                [20, 30, 40, 50] by default.
    :param gd_function: {0: gradient descent with ReLU,
                else: gradient descent with sigmoid function}
    :return:
    """
    if gd_function == 0:
        gradient_descent = gradient_descent_with_relu
        title = 'Accuracy over different sizes of hidden layer with ReLU'
    else:
        gradient_descent = gradient_descent_with_sigmoid_function
        title = 'Accuracy over different sizes ' \
                'of hidden layer with sigmoid function'

    accuracy_array = np.empty(len(n_units))

    for i in range(len(n_units)):
        _, accuracy_array[i] = gradient_descent(
            x1,
            y1,
            N=N,
            D=D,
            K=K,
            n_units=n_units[i],
        )

        figure3_1 = plt.figure()
        plt.plot(n_units, accuracy_array * 100, 'r+-')
        plt.title(title)
        plt.xlabel('Size of the hidden layer')
        plt.ylabel('accuracy percentage')
        plt.show()


def exercise1_2():
    x1, y1 = pre_train()

    n_units = list(range(20, 60, 10))
    accuracy_over_different_sizes_of_hidden_layer(
        x1=x1, y1=y1, N=100, D=2, K=3,
        n_units=n_units, gd_function=0
    )
    n_units = list(range(10, 70, 10))
    accuracy_over_different_sizes_of_hidden_layer(
        x1=x1, y1=y1, N=100, D=2, K=3,
        n_units=n_units, gd_function=0
    )
    n_units = list(range(10, 61, 1))
    accuracy_over_different_sizes_of_hidden_layer(
        x1=x1, y1=y1, N=100, D=2, K=3,
        n_units=n_units, gd_function=0
    )


def exercise1_3_1():
    # generate hyper-parameters ranging from 0.1 to 10^-9
    rate_list = [math.pow(10, -n) for n in range(1, 1 + 9)]

    x1, y1 = pre_train()

    loss_values_over_iterations_with_different_learning_rates(
        x1=x1, y1=y1, N=100, D=2, K=3,
        learning_rates=rate_list,
        gd_function=1,
    )


def exercise1_3_2():
    x1, y1 = pre_train()

    n_units = list(range(20, 60, 10))
    accuracy_over_different_sizes_of_hidden_layer(
        x1=x1, y1=y1, N=100, D=2, K=3,
        n_units=n_units, gd_function=1
    )
    n_units = list(range(10, 70, 10))
    accuracy_over_different_sizes_of_hidden_layer(
        x1=x1, y1=y1, N=100, D=2, K=3,
        n_units=n_units, gd_function=1
    )
    n_units = list(range(10, 61, 1))
    accuracy_over_different_sizes_of_hidden_layer(
        x1=x1, y1=y1, N=100, D=2, K=3,
        n_units=n_units, gd_function=1
    )


if __name__ == '__main__':
    exercise1_1()
    # exercise1_2()
    # exercise1_3_1()
    # exercise1_3_2()

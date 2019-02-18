import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))  # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# generate hyper-parameters ranging from 0.1 to 10^-9
N_RATES = 9
learning_rates = [math.pow(10, -n) for n in range(1, 1 + N_RATES)]

# define loss value record
N_ITERATIONS = 100_000
loss_array = np.zeros((N_ITERATIONS, N_RATES))

dtype = torch.FloatTensor
# augment input vectors
x1 = Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
el1 = torch.cat((torch.ones(100), torch.zeros(200)), 0)
el2 = torch.cat((torch.zeros(100), torch.ones(100), torch.zeros(100)), 0)
el3 = torch.cat((torch.zeros(200), torch.ones(100)), 0)
yy = torch.stack((el1, el2, el3), 1)
y1 = Variable(torch.stack((el1, el2, el3), 1), requires_grad=False)
h = 50  # size of hidden layer

for j in range(N_RATES):
    # parameters
    w1 = Variable(0.5 * torch.randn(D, h), requires_grad=True)
    b1 = Variable(torch.randn(1, h), requires_grad=True)
    w2 = Variable(0.5 * torch.randn(h, K), requires_grad=True)
    b2 = Variable(torch.randn((1, K)), requires_grad=True)

    # run with different learning rates
    # gradient descent loop
    for i in range(N_ITERATIONS):

        # Forward pass
        hout = x1.mm(w1) + b1
        h_relu = hout.clamp(min=0)
        output = h_relu.mm(w2) + b2  # output of NN
        # scores = output.clamp(min=0)
        scores = output

        # compute the loss
        loss = (scores - y1).pow(2).sum()
        loss_array[i, j] = loss

        loss.backward()

        if i % 10_000 == 0:
            print(j, i, loss)

        # perform a parameter update
        w2.data -= learning_rates[j] * w2.grad.data
        w1.data -= learning_rates[j] * w1.grad.data
        b2.data -= learning_rates[j] * b2.grad.data
        b1.data -= learning_rates[j] * b1.grad.data

        # Manually zero the gradient after the backward pass
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b2.grad.data.zero_()
        b1.grad.data.zero_()

# print the first 10 loss values
for i in range(10):
    for j in range(N_RATES):
        print('{0: 10.3f}\t'.format(loss_array[i, j]), end='')
    print('')

# plot the loss values
figure2 = plt.figure()
plot_x = [n for n in range(1, N_ITERATIONS + 1)]
for i in range(N_RATES):
    plt.plot(plot_x, loss_array[:, i])

plt.ylim(0, 200.0)
plt.xlabel('iteration number')
plt.ylabel('loss value')
legend = ['learning rate is {}'.format(rate) for rate in learning_rates]
plt.legend(legend)
plt.show()

# evaluate training set accuracy
desired_class = torch.cat((torch.zeros(100), torch.ones(100), torch.add(torch.ones(100), 1)), 0)
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

import Exercise1 as e1

#####################################################################
# 1. generate required dataset

# number of points in each class
SIZE = 150
# number of dimension
N_DIMENSION = 2
# number of classes
N_CLASSES = 2

# (150 * 2 x 2) matrix, used to store all data/point, of which
# the first 150 points belong to the first class,
# while the second 150 points belong to the second class.
x = np.zeros((SIZE * N_CLASSES, N_DIMENSION))
# data label/class
y = np.zeros(SIZE * N_CLASSES, dtype='uint8')

for i in range(N_CLASSES):
    # radius of the circle in the figure
    # the radius look like under the uniform distribution
    if i == 0:
        # radius of the first class, from 0 to 1
        radius = np.random.uniform(0.0, 1, SIZE)
    else:
        # radius of the second class, from about 1.5 to 2
        radius = np.random.uniform(1.5, 2, SIZE)

    # angle degree vector
    # tx = evenly generated base degree +/- [0, pi/300]
    t = np.linspace(0, 2 * np.pi, SIZE) \
        + np.random.uniform(-1, 1, SIZE) * np.pi / (SIZE * N_CLASSES)

    index_range = range(SIZE * i, SIZE * (i + 1))

    # to generate the 300 coordinates
    # x[0: 150, :] --> class 1
    # x[150: 300, :] --> class 2
    x[SIZE * i: SIZE * (i + 1), :] = \
        np.c_[radius * np.sin(t), radius * np.cos(t)]

    # y[0: 150] = 0, y[150, 300] = 1
    y[SIZE * i: SIZE * (i + 1)] = i

# now  visualize the data set
figure1 = plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

#####################################################################
# 2. train on the data
dtype = torch.FloatTensor
x1 = Variable(torch.from_numpy(x).type(dtype), requires_grad=False)
el1 = torch.cat((torch.ones(SIZE), torch.zeros(SIZE)), 0)
el2 = torch.cat((torch.zeros(SIZE), torch.ones(SIZE)), 0)
y1 = Variable(torch.stack((el1, el2), 1), requires_grad=False)

n_unit = [20, 30, 40, 50, 60]
e1.accuracy_over_different_sizes_of_hidden_layer(
    x1, y1, N=150, D=2, K=2,
    n_units=n_unit, gd_function=1
)

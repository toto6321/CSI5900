# Assignment 2 of Tianhuan Tu
[TOC]

## Exercise 1
1. The plot is in the .ipynb file. We could conclude from the chart that... 
    * Local optimal point could not be reached, when the learning rate is much bigger than 0.001.
    * The learning is kind of slow when the learning rate is much smaller than 0.1e-6
    * The best learning rate is about 0.1e-3, which is 0.0001.

2. The best learning rate is about 0.0001. The plot could be found in the .ipynb file.
    * It seems that the accuracy first increase and then decrease when the size of the hidden layers increases, 
        and it has the best accuracy when the size is 40, from the chart.
    * However, the plots will be totally different when repeating the test.
    * Extend the range of the size of the hidden layer and test again.
    * Decrease the step of the size range and test again.

3. Repeat the step above with the ReLU replaced by sigmoid function.
    1. It seems to have best performance with the learning rate of 0.0001 again. However, it becomes kind of unstable. 
        I would prefer to the 0.00001.
    2. Basically, we could conclude that it has the best accuracy when the size of the hidden layer is 50, 
        though it behaves really weired with the size increasing.
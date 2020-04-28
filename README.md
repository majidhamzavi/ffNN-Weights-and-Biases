# ffNN-Weights-and-Biases
Feed Forward Neural Network- generate and optimize random Weights and Biases

In this repository, I will explain how to create random weights and biases for a 2-layer neural network (NN). Based on how many layers and neurons you might have in NN, you need to generate random weights and biases and implement ffNN explained in the previous repository (https://github.com/majidhamzavi/feedforwardNN). 

The same as first example, I am taking 2 hidden layers with 512 neurons. Therefore, weights and biases dimensions will be as follows:
I) x[60,000 * 784] --> w0[784 * 512], b0[512 * 1], w1[512 * 512], b1[512 * 1], w2[512 * 10], b2[10 * 1] --> out(y)[60,000 * 10].
Also, it is beneficial to implement one-hot encoder for the 'y' and its dimensions would be [60,000 * 10] instead of [60,000 * 1]. 

II) The trick is here: fist, with current weight and biases, we calculate the error (mean_squared_error). Then manipulate the weight and biases and one more time we obtain the error. If error decreased, we update weights and Biases, otherwise, previous weights and biases will be kept. We continue this algorithm up to a max_iteration.

Notes: 
1- It is easy to modify weights and biases by adding a small random number to each element.
2- To decrease the running time, we can randomly choose a batch of x_train in each iteration. 

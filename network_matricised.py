# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            X, Y = zip(*training_data)
            # reshape is to remove an extra depth dimension of 1 which appears
            mini_batches_X = [
                np.array(X[k:k+mini_batch_size]).reshape((self.sizes[0], mini_batch_size)) for k in range(0, n, mini_batch_size)]
            mini_batches_Y = [
                np.array(Y[k:k+mini_batch_size]).reshape((self.sizes[-1], mini_batch_size)).transpose()
                for k in range(0, n, mini_batch_size)]
            '''
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            '''
            for mini_batch_x, mini_batch_y in zip(mini_batches_X, mini_batches_Y):
                self.update_mini_batch(mini_batch_x, mini_batch_y, eta, mini_batch_size)
            
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch_x, mini_batch_y, eta, mini_batch_size):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        '''
        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch_x, mini_batch_y, mini_batch_size)
        #print('db', delta_nabla_b[0].shape)
        # average the errors between x's
        #print('w',self.weights[0].shape)
        self.weights = [w-(eta/mini_batch_size)*nw
                        for w, nw in zip(self.weights, delta_nabla_w)]
        #print('w', self.weights[0].shape)
        #print('b', self.biases[0].shape)
        self.biases = [b-(eta/mini_batch_size)*nb
                       for b, nb in zip(self.biases, delta_nabla_b)]
        #print('b', self.biases[0].shape)

    def backprop(self, x, y, mini_batch_size):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #print(nabla_w[-1].shape)
        # feedforward
        '''
        activation = x
        '''
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        #print("b:", self.biases)
        for b, w in zip(self.biases, self.weights):
            #print("w:", w.shape, "\na:", activations[-1].shape)
            #print("w*a:", np.dot(w, activations[-1]).shape)
            #print("b:", np.broadcast_to(b, (len(w), mini_batch_size)).shape)
            z = np.dot(w, activations[-1]) + np.broadcast_to(b, (len(w), mini_batch_size))
            zs.append(z)
            '''
            activation = sigmoid(z)
            '''
            activations.append(sigmoid(zs[-1]))
        # backward pass
        #print("output_activations", activations[-1].shape)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        #print("delta_last_layer", delta.shape)
        nabla_b[-1] = np.sum(delta, axis=1).reshape((10, 1))
        #print("activations_second_layer", activations[-2].shape)
        #print("^transpose", activations[-2].transpose().shape)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #print("nabla_w_last_layer", nabla_w[-1].shape)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #print("delta_first_layer", delta.shape)
            nabla_b[-l] = np.sum(delta, axis=1).reshape((30, 1))
            #print("activations_input_layer", activations[-l-1].shape)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            #print("nabla_b_first_layer", nabla_b[-l].shape)
            #print("nabla_w_first_layer", nabla_w[-l].shape)
        #raise
        #print('dbi', nabla_b[0].shape)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

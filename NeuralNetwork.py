# *******************************
# ********** LIBRARIES **********
# *******************************
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from NeuralLayer import NeuralLayer

# HELP RESOURCES:
# https://www.python-course.eu/neural_networks_with_python_numpy.php
# https://github.com/pangolulu/rnn-from-scratch
# https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# https://github.com/pangolulu/neural-network-from-scratch
# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

# Good Resources:
# https://github.com/stephencwelch/Neural-Networks-Demystified
# https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src
# https://colab.research.google.com/drive/1vm1EAZ7lLRooZHqHTfUlul5ZXpGmb_SZ#scrollTo=48D6_obVPuZS
# https://matrices.io/deep-neural-network-from-scratch/

# https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py


class NeuralNetwork:
    """
    Description:
        -Creates a new neural network as an list of neural layers,
        based on topology.

    Attributes:
        -network (NeuralLayer []): list of neural layers that
        represents the neural network

    Args:
        -topology (int []): an input list that represents the
        topology of this neural network. Each element is the
        number of neurons in that layer. Ex:
            [2, 3, 4, 2]: 2 elements in the 1st layer, 3 in the
            2nd layer and so on.
    """
    def __init__(self, topology):
        # The first layer is not considered as a real layer, but as an input layer.
        self.network = [None]

        for l in range(0, len(topology) - 1):
            self.network.append(NeuralLayer(topology[l + 1], topology[l]))

    def feedforward(self, input_x):
        """
        Description:
            -Performs y[L] = sigm(W[L] * x[L-1] + B[L]), where L is the actual layer,
            over the neural network with an initial input input_x.

        Args:
            -input_x (double[input neurons][1]): initial input of the neural network.

        Note:
            -This method is used to make the actual predictions of the neural network.
        """
        for l in range(1, len(self.network)):
            y = sigm(np.dot(self.network[l].W, input_x) + self.network[l].B)
            input_x = y

        return y

    def backpropagation(self, input_x, target_y, rate):
        """
        Description:
            -Backpropagates the error over the previous layers. After that, it
            performs Gradient Descent algorithm to adjust weights and biases.

        Args:
            -input_x (double[][1]): initial input of the neural network.
            -target_y (double[][1]): target value that is wanted to be achieved.
            -rate (double): the learning rate of backpropagation.
        """
        # Store all activations of each layer (y)
        activations = [sigm(input_x)]

        # Store all weighted sums of each layer (s).
        weighted_sum = [input_x]

        # **** FORWARD PASS ****
        for l in range(1, len(self.network)):
            # Calculate the activation and store it
            y = sigm(np.dot(self.network[l].W, input_x) + self.network[l].B)
            activations.append(y)

            # Calculate the weighted sum and store it
            s = np.dot(self.network[l].W, input_x) + self.network[l].B
            weighted_sum.append(s)

            input_x = y

        # **** BACKWARD PASS ****
        # Store layers' errors
        errors = []

        # Calculate delta for the last layer
        layer = len(self.network) - 1   # last layer
        errors.insert(0, ms_error_prime(target_y, activations[layer]) * sigm_prime(weighted_sum[layer]))

        for l in reversed(range(1, len(self.network))):
            y = activations[l]
            s = weighted_sum[l - 1]

            # Retropropagate the error over the network
            delta = np.dot(np.transpose(self.network[l].W), errors[0]) * sigm_prime(s)

            # **** GRADIENT DESCENT ****
            # Update Weights
            self.network[l].W = self.network[l].W + rate * np.transpose(np.dot(delta, np.transpose(y)))

            # Update Biases
            self.network[l].B = self.network[l].B + rate * errors[0]

            # Update the error
            errors.insert(0, delta)

    def train(self, input_x, target_y, rate, n_iterations):
        """
        Description:
            Trains the network using the given inputs.

        Args:
            -input_x (double[][1]): initial input of the neural network.
            -target_y (double[][1]): target value that is wanted to be achieved.
            -rate (double): the learning rate of backpropagation.
            -n_iterations (int): the number of iterations of the training.
        """
        # Loss function
        loss = []

        for i in range(n_iterations):
            # Backpropagation
            nN.backpropagation(np.transpose(input_x), np.transpose(target_y), rate)

            # Make Prediction
            Yp = nN.feedforward(np.transpose(input_x))

            # Losses results
            loss.append(ms_error(Yp, np.transpose(target_y)))

            # Plotting settings
            plt.plot(range(len(loss)), loss, color='#EE6666')
            plt.title('Loss function')
            ax = plt.axes(axisbg='#E6E6E6')
            ax.set_axisbelow(True)
            plt.grid(color='w', linestyle='solid')

            # Plot in real time
            plt.pause(0.05)

        plt.show()


# ************************************************
# ************* ACTIVATION FUNCTIONS *************
# ************************************************
def sigm(s):
    """
    Description:
        -Calculates the output of a given value using
        the sigmoid function.

    Args:
        -s (double): a given input number.
    """
    return 1.0 / (1.0 + np.e ** (-s))


def sigm_prime(s):
    """
    Description:
        -Calculates the output of a given value using
        the sigmoid derivative function.

    Args:
        -s (double): a given input number.
    """
    return sigm(s)*(1 - sigm(s))


# ************************************************
# **************** COST FUNCTIONS ****************
# ************************************************
def ms_error(y_p, y_r):
    """
    Description:
        -Calculates the mean square error of a given
        set of values.

    Args:
        -y_p (double [output neurons][1]): Predicted value.
        -y_r (double [output neurons][1]): Real value
    """
    return np.mean(y_p - y_r)**2


def ms_error_prime(y_p, y_r):
    """
    Description:
        -Derivative of mean square error.

    Args:
        -y_p (double [output neurons][1]): Predicted value.
        -y_r (double [output neurons][1]): Real value
    """
    return 2 * np.mean(y_p - y_r)


# **************************************************
# ********************** MAIN **********************
# **************************************************
n = 500
input_variables = 3
output_variables = 2

# Generate Dataset
X = np.random.rand(500, input_variables)
Y = np.random.rand(500, output_variables)

# Create the network
topology = [input_variables, 4, 8, output_variables]
nN = NeuralNetwork(topology)

# Train the network
rate = 0.05
iterations = 1200
nN.train(X, Y, rate, iterations)


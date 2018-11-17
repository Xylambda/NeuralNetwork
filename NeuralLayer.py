# *******************************
# ********** LIBRARIES **********
# *******************************
import numpy as np


class NeuralLayer:
    """
    Description:
        -Creates a new layer based on n_this and n_prev.
        -The biases matrix has the shape: [n_this, 1]
        -The weights matrix has the shape: [n_this, n_prev]

    Attributes:
        -B (double [n_this][n_prev]): The biases matrix of this layer.
        -W (double [n_this][1]): The weights matrix of this layer.

    Args:
        -n_this (int): number of neurons of this layer.
        -n_prev (int): number of neurons of the previous layer.
    """

    def __init__(self, n_this, n_prev):
        self.B = np.random.rand(n_this, 1)
        self.W = np.random.rand(n_this, n_prev)
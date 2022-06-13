import numpy as np
from alternativelib.layer import layer_type

class FullyConnected():
    def __init__(self, in_dims, out_dims, batch_size):
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.layer_type = layer_type.dense
        self.batch_size = batch_size


        self._weights = np.random.normal(0, 1, size=(out_dims, in_dims)) / 10
        self._bias = np.random.normal(0, 1, size=(out_dims, 1)) / 10
        self._old_gradient_weight = None
        self._old_gradient_bias = None

        self._gradient_weight = np.zeros((batch_size, out_dims, in_dims))
        self._gradient_bias = np.zeros((batch_size, out_dims, 1))
        self._output = np.zeros((batch_size, out_dims, 1))
        self._input = np.zeros((batch_size, in_dims, 1))

    def __call__(self, input, test=False):

        self._input = input.copy()

        if test:
            z = np.dot(self._weights, self._input[0]) + self._bias
            return np.expand_dims(z, axis=0)

        for b in range(self.batch_size):
            z = np.dot(self._weights, self._input[b]) + self._bias
            self._output[b] = z

        return self._output.copy()

    def _backward(self, delta):  # Do i need this ?
        """

        :param delta: Bxoutx1
        :return:
        """
        for b in range(self.batch_size):
            self._gradient_weight[b] = np.dot(delta[b], self._input[b].T)  # in x 1 @ 1 x out
            # w_-1 (in x out) = a_1 (in x 1) @ delta.T (1 x out)
            self._gradient_bias[b] = delta[b].copy()

class Flatten():
    def __init__(self, batch_size):
        self.layer_type = layer_type.flatten
        self._input = None
        self._output = None
        self._input_shape = None
        self.batch_size = batch_size

    def __call__(self, input, test=False):

        self._input = input

        self._input_shape = input.shape

        a = input.reshape(self._input_shape[0], -1, 1)

        self._output = a
        return a

    def _backward(self, delta):
        return delta.reshape(self._input_shape)
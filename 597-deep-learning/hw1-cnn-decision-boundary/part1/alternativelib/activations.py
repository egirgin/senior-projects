import numpy as np
from alternativelib.layer import layer_type

class relu():
    def __init__(self):
        self.layer_type = layer_type.relu
        self._input = None
        self._output = None


    def __call__(self, input, test=None):

        self._input = input.copy()

        a = np.maximum(self._input, 0)

        self._output = a.copy()

        return a

    def _backward(self):

        grads = np.zeros_like(self._input)

        grads[self._input > 0] = 1

        return grads


class linear():
    def __init__(self):
        self.layer_type = layer_type.linear
        self._input = None
        self._output = None

    def __call__(self, input, test=None):
        self._input = input.copy()
        self._output = self._input
        return self._input

    def _backward(self):
        return np.ones_like(self._input)

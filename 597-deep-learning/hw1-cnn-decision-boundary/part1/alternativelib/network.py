import numpy as np
from alternativelib.layer import layer_type

class Network:
    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.batch_size = None

    def add_layer(self, layer):
        self.layers.append(layer)
        self.num_layers += 1

    def __call__(self, input, test=False):

        batch_size = input.shape[0]
        if self.batch_size == None:
            self.batch_size = batch_size

        x = input
        for i, layer in enumerate(self.layers):

            x = layer(x, test)

        return x

    def backwards(self, error_deriv):
        """
        :param error_deriv: Bx10x1
        :return:
        """

        delta = error_deriv

        self.layers[-2]._backward(delta)

        for layer_id in reversed(range(self.num_layers - 2)):

            if self.layers[layer_id].layer_type == layer_type.dense:
                #print(self.layers[layer_id]._weights.shape)
                new_delta = np.zeros((self.batch_size, self.layers[layer_id].out_dims, 1))
                for b in range(self.batch_size):
                    new_delta[b] = self.layers[layer_id + 2]._weights.T @ delta[b]

                delta = new_delta
                delta = np.multiply(delta, self.layers[layer_id+1]._backward())

                self.layers[layer_id]._backward(delta)

            if self.layers[layer_id].layer_type == layer_type.flatten:
                delta = self.layers[layer_id]._backward(delta)

            if self.layers[layer_id].layer_type == layer_type.convolutional:
                delta = self.layers[layer_id]._backward(delta)

            if self.layers[layer_id].layer_type == layer_type.maxpool:
                delta = self.layers[layer_id]._backward(delta)
                delta = np.multiply(delta, self.layers[layer_id-1]._backward())





    def update(self, lr=0.001, momentum=0.9):
        for layer in self.layers:
            if layer.layer_type == layer_type.dense or layer.layer_type == layer_type.convolutional:
                if layer._old_gradient_weight is None:
                    layer._old_gradient_weight = -np.mean(layer._gradient_weight, axis=0)
                    layer._old_gradient_bias = -np.mean(layer._gradient_bias, axis=0)

                # calculate gradient with momentum
                new_change_weight = -lr * np.mean(layer._gradient_weight, axis=0) + momentum * layer._old_gradient_weight
                new_change_bias = -lr * np.mean(layer._gradient_bias, axis=0) + momentum * layer._old_gradient_bias

                # update weights
                layer._weights += new_change_weight
                layer._bias += new_change_bias

                # calculate gradient with momentum
                #new_change_weight = -lr * np.mean(layer._gradient_weight, axis=0) + momentum * layer._old_gradient_weight
                #new_change_bias = -lr * np.mean(layer._gradient_bias, axis=0) + momentum * layer._old_gradient_bias

                # update weights
                #layer._weights += new_change_weight
                #layer._bias += new_change_bias

                # update momentum coeff
                layer._old_gradient_weight = new_change_weight
                layer._old_gradient_bias = new_change_bias
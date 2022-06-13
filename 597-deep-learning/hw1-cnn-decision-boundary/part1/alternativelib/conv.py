from alternativelib.layer import layer_type
import numpy as np

class conv2d():
    def __init__(self, in_channels, out_channels, filter_size, batch_size, stride=1, padding=0, ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.batch_size = batch_size
        self.layer_type = layer_type.convolutional

        self.in_h = None
        self.in_w = None
        self.out_h = None
        self.out_w = None

        self._weights = np.random.normal(0, 1, size=(self.in_channels, self.out_channels, filter_size, filter_size)) / 10

        self._bias = None

        self._gradient_weight = np.zeros((self.batch_size, self.in_channels, self.out_channels, filter_size, filter_size))
        #self._gradient_bias = np.zeros_like(self._bias)
        self._output = None
        self._input = None

        self._old_gradient_weight = None
        self._old_gradient_bias = None

    def _set_bias(self):
        if self._bias is None:
            self._bias = np.random.normal(0, 1, size=(self.out_channels, self.out_h, self.out_w)) / 10
            self._gradient_bias = np.zeros((self.batch_size, self.out_channels, self.out_h, self.out_w))

    def _cross_corr(self, input, kernel):

        k_h, k_w = kernel.shape

        out_h = (input.shape[0] - kernel.shape[0]) / self.stride + 1
        out_w = (input.shape[1] - kernel.shape[1]) / self.stride + 1
        out_h = int(out_h)
        out_w = int(out_w)

        a = np.zeros((out_h, out_w))

        for row_id in range(out_h):
            for col_id in range(out_w):
                a[row_id, col_id] = np.sum(np.multiply(
                    input[row_id:row_id+k_h, col_id:col_id+k_w],
                    kernel
                ))

        return a

    def _dcross_corr(self, delta, kernel):

        pad = self.filter_size - 1

        dy = np.pad(delta, [pad, pad], mode="constant", constant_values=0)

        return self._cross_corr(dy, np.flip(kernel))

    def __call__(self, input, test=False):
        """
        :param input: BxCxHxW
        :return: BxCxHxW
        """
        self._input = input.copy()
        batch_size, in_c, in_h, in_w = input.shape
        self.in_h = in_h
        self.in_w = in_w

        out_h = (in_h - self.filter_size + 2 * self.padding) / self.stride + 1
        out_w = (in_w - self.filter_size + 2 * self.padding) / self.stride + 1
        out_h = int(out_h)
        out_w = int(out_w)

        self.out_h = out_h
        self.out_w = out_w

        self._set_bias()

        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for batch_id in range(batch_size):
            for out_c in range(self.out_channels):
                channel = np.zeros((out_h, out_w))
                for in_c in range(self.in_channels):
                    channel += self._cross_corr(input[batch_id, in_c], self._weights[in_c, out_c])

                output[batch_id, out_c] = channel.copy() + self._bias[out_c]

        self._output = output

        return output


    def _backward(self, delta):

        next_delta = np.zeros_like(self._input)

        _, c_in, h_in, w_in = self._input.shape

        _, c_out, h_out, w_out = delta.shape

        for b in range(self.batch_size):
            for c in range(c_in):
                channel = np.zeros((h_in, w_in))
                for d in range(c_out):
                    deriv = self._cross_corr(self._input[b, c], delta[b, d])
                    #print(d)
                    self._gradient_weight[b, c, d] = deriv.copy()
                    channel += self._dcross_corr(delta[b, d], self._weights[c, d]) # TODO maybe take average instead of sum ?

                next_delta[b, c] = channel.copy()

        self._gradient_bias = delta.copy()

        return next_delta


class MaxPool2d():
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self._input = None
        self._output = None
        self._gradients = None
        self.layer_type = layer_type.maxpool
        self.batch_size = None

    def __call__(self, input, test=False):
        b, c, h, w = input.shape
        self.batch_size = b

        out_h = h/self.kernel_size
        out_w = h/self.kernel_size
        out_h = int(out_h)
        out_w = int(out_w)

        a = np.zeros((b, c, out_h, out_w))

        grads = np.zeros((b, c, h, w))

        for batch_id in range(b):
            for channel_id in range(c):
                for row_id in range(out_h):
                    for col_id in range(out_w):
                        start_row = row_id*self.kernel_size
                        start_col = col_id*self.kernel_size
                        receptive_field = input[batch_id, channel_id, start_row:start_row+self.kernel_size, start_col:start_col+self.kernel_size]

                        max_val = np.max(receptive_field)
                        max_idx = np.argwhere(receptive_field == max_val)[0]

                        a[batch_id, channel_id, row_id, col_id] = max_val
                        grads[batch_id, channel_id, start_row+max_idx[0], start_col+max_idx[1]] = 1

        self._output = a
        self._gradients = grads

        return a

    def _backward(self, delta):
        upsampled = np.repeat(delta, self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)

        return np.multiply(upsampled, self._gradients)




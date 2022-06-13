import numpy as np

class CrossEntropy():
    def __init__(self,batch_size, use_softmax=False):
        self.use_softmax = use_softmax
        self.batch_size = batch_size

    def softmax(self, input):
        exps = np.exp(input - input.max()) # minus max is for numerical stability
        a = exps / np.sum(exps)
        return a

    def __call__(self, pred, target):
        pred_copy = pred.copy()

        outputs = np.zeros_like(target)

        for b in range(self.batch_size):

            softmax_i = self.softmax(pred_copy[b])

            loss = -np.log(softmax_i[target[b]])
            outputs[b] = loss
        return np.mean(outputs)


    def _backward(self, pred, target):

        grads = np.zeros_like(pred)

        for b in range(self.batch_size):

            if self.use_softmax:
                softmax = self.softmax(pred[b].copy())

                softmax[target[b]] -= 1

                grads[b] = softmax

            else:
                grads[b, target[b]] = -1/pred[b, target[b]]

        return grads

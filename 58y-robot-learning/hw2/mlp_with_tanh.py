import numpy as np
np.random.seed(42)

def tanh(x):
    if (type(x) == int) or (type(x) == float):
        if x > 30:
            return 1.0
        elif x < -30:
            return -1.0
        else:
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    elif type(x) == np.ndarray:
        act = np.zeros_like(x)
        act = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        act[x>30] = 1.0
        act[x<-30] = -1.0
    return act

def dtanh(x):
    return 1-np.power(np.tanh(x), 2)

class Network:
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        self.W = [np.random.randn(input_dim, hidden_dim)*((2/(input_dim+hidden_dim))**0.5)]
        self.b = [np.zeros(hidden_dim)]
        self.gradW = [np.zeros((input_dim, hidden_dim))]
        self.gradb = [np.zeros(hidden_dim)]
        for i in range(num_layers):
            self.W.append(np.random.randn(hidden_dim, hidden_dim)*((1/hidden_dim)**0.5))
            self.b.append(np.zeros(hidden_dim))
            self.gradW.append(np.zeros((hidden_dim, hidden_dim)))
            self.gradb.append(np.zeros(hidden_dim))
        self.W.append(np.random.randn(hidden_dim, output_dim)*((2/(hidden_dim+output_dim))**0.5))
        self.b.append(np.zeros(output_dim))
        self.gradW.append(np.zeros((hidden_dim, output_dim)))
        self.gradb.append(np.zeros(output_dim))
        self.activations = []
        self.z = []

    def forward(self, x):
        self.activations = [x.copy()]
        self.z = []
        for i in range(len(self.W)-1):

            x = x @ self.W[i] + self.b[i]
            self.z.append(x)

            x = np.tanh(x)
            # x = np.maximum(x @ self.W[i] + self.b[i], 0)
            self.activations.append(x.copy())

        x = x @ self.W[-1] + self.b[-1]
        return x

    def backward(self, error):
        """
        :param error: 2x1
        :return:
        """
        delta = -error # minus comes from derivative

        self.gradW[-1] = self.activations[-1].reshape(-1, 1) @ delta.reshape(-1, 1).T # hidx1 @ 1x2
        self.gradb[-1] = delta


        for layer_id in reversed(range(len(self.W) - 1)):
            # 3, 2, 1, 0

            delta = self.W[layer_id+1] @ delta # hidx2 @ 2x1
            delta = np.multiply(delta, dtanh(self.z[layer_id])) # hidx1


            self.gradW[layer_id] = self.activations[layer_id].reshape(-1, 1) @ delta.reshape(-1, 1).T #hidx1 @ 1xhid
            self.gradb[layer_id] = delta


    def update(self, lr):
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - lr * self.gradW[i]
            self.b[i] = self.b[i] - lr * self.gradb[i]

    def __repr__(self):
        return "(" + ", ".join([str(x.shape[0]) for x in self.W]) + ", " + str(self.W[-1].shape[1]) + ")"

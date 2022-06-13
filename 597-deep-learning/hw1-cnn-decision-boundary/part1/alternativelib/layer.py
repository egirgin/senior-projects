import enum

class layer_type(enum.Enum):
    dense = 1
    convolutional = 2
    relu = 3
    tanh = 4
    sigmoid = 5
    softmax = 6
    linear = 7
    loss = 8
    maxpool = 9
    flatten = 10

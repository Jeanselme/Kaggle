import csv
import sys
import numpy as np
import itertools as it
from  scipy import ndimage

############################################################################
# Helpers
############################################################################


def loop_2_1(r):
    return it.product(r, r)


def loop_2_2(r1, r2):
    return it.product(r1, r2)


def loop_3_3(r1, r2, r3):
    return loop_2_2(r1, loop_2_2(r2, r3))


############################################################################
# Layers
############################################################################


class Layer:

    def __init__(self, values_shape):
        self.values_shape = values_shape
        self.depth = values_shape[0]
        self.width = values_shape[1]
        self.network = None

    def setup(self, network, prev_layer, next_layer):
        self.network = network
        self.prev_layer = prev_layer
        self.next_layer = next_layer

    def forward(self, value, label):
        return value

    def backward(self, score, step):
        return score

    def get_desc(self):
        return "Layer " + str(self.values_shape)


class ConvLayer(Layer):

    class Filter:

        def __init__(self, depth, size):
            self.kernel = 0.1 * np.random.randn(depth, size, size)
            self.kernel_grad = np.zeros(self.kernel.shape)
            self.bias = 0.1 * np.random.randn()
            self.bias_grad = 0.0
            self.size = size

        def clear_gradient(self):
            self.kernel_grad.fill(0)
            self.bias_grad = 0

    def __init__(self, values_shape, filter_size, padding):
        Layer.__init__(self, values_shape)
        self.filter_size = filter_size
        self.padding = padding
        self.filters = []
        self.temp = None

    def setup(self, network, previous_layer, next_layer):
        Layer.setup(self, network, previous_layer, next_layer)
        for i in range(next_layer.depth):
            filt = ConvLayer.Filter(self.depth, self.filter_size)
            self.filters.append(filt)

    def forward(self, value, label):
        self.temp = value
        new_width = self.width + 2 * self.padding
        out_depth = len(self.filters)
        out_width = (new_width - self.filter_size) + 1
        result = np.zeros([out_depth, out_width, out_width])

        for k in range(out_depth):
            new_width = self.width + 2 * self.padding
            new_input = np.zeros([self.depth, new_width, new_width])
            end = new_width - self.padding
            new_input[:, self.padding:end, self.padding:end] = value

            d_width = (new_width - self.filter_size) + 1
            output = np.zeros([d_width, d_width])

            for (i, j) in loop_2_1(range(d_width)):
                shift_i = i + self.filter_size
                shift_j = j + self.filter_size
                region = new_input[:, i:shift_i, j:shift_j]
                kernel = self.filters[k].kernel
                bias = self.filters[k].bias
                output[i, j] = np.sum(region * kernel) + bias
            result[k, :, :] = output
        return result

    def backward(self, score, step):
        new_width = self.width + 2 * self.padding
        new_input = np.zeros([self.depth, new_width, new_width])
        end = new_width - self.padding

        new_input[:, self.padding:end, self.padding:end] = self.temp
        output = np.zeros(new_input.shape)

        for k in range(len(self.filters)):
            filt = self.filters[k]
            filt.clear_gradient()

            for (i, j) in loop_2_1(range(new_width - filt.size + 1)):
                (shift_i, shift_j) = (i + filt.size, j + filt.size)
                region = new_input[:, i:shift_i, j:shift_j]
                out_filt = score[k, i, j] * filt.kernel
                output[:, i:shift_i, j:shift_j] += out_filt
                filt.kernel_grad += score[k, i, j] * region
                filt.bias_grad += score[k, i, j]

        reg = self.network.lambda_reg
        for filt in self.filters:
            filt.kernel -= filt.kernel_grad * step + reg * filt.kernel
            filt.bias -= filt.bias_grad * step
        return output[:, self.padding:end, self.padding:end]

    def get_desc(self):
        return "ConvLayer " + str(self.values_shape) + " " + \
               str(self.filter_size) + " " + str(self.padding)


class PoolLayer(Layer):

    def __init__(self, values_shape):
        Layer.__init__(self, values_shape)
        self.pool_index = None

    def forward(self, value, label):
        output = np.zeros([self.depth, self.width // 2, self.width // 2])
        self.pool_index = np.zeros(output.shape)

        r1 = range(self.depth)
        r2 = range(0, self.width, 2)
        for (d, (i, j)) in loop_3_3(r1, r2, r2):
            region = value[d, i:i+2, j:j+2]
            output[d, i // 2, j // 2] = np.max(region)
            self.pool_index[d, i // 2, j // 2] = np.argmax(region)
        return output

    def backward(self, score, step):
        r0 = range(score.shape[0])
        r1 = range(score.shape[1])
        r2 = range(score.shape[2])

        gradient = np.zeros(self.values_shape)
        for (k, (i, j)) in loop_3_3(r0, r1, r2):
            (a, b) = (2 * i, 2 * i + 2)
            (c, d) = (2 * j, 2 * j + 2)
            e = int(self.pool_index[k, i, j] // 2)
            f = int(self.pool_index[k, i, j] % 2)
            gradient[k, a:b, c:d][e, f] = score[k, i, j]
        return gradient

    def get_desc(self):
        return "PoolLayer " + str(self.values_shape)


class ReLULayer(Layer):

    def __init__(self, values_shape):
        Layer.__init__(self, values_shape)
        self.acts = None

    def forward(self, value, label):
        self.acts = value > 0
        return self.acts * value

    def backward(self, score, step):
        return self.acts * score

    def get_desc(self):
        return "ReLuLayer " + str(self.values_shape)


class SoftLayer(Layer):

    def __init__(self, values_shape):
        Layer.__init__(self, values_shape)
        self.label = -1

    def forward(self, value, label):
        self.label = label
        score = (value - np.max(value)) * 0.05
        return score

    def backward(self, score, step):
        gradient = np.exp(score) / np.sum(np.exp(score))
        gradient[int(self.label)] -= 1.0
        return gradient * 0.05

    def get_desc(self):
        return "SoftLayer " + str(self.values_shape)


############################################################################
# Network
############################################################################


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.lambda_reg = 0.01
        self.step_init = 0.001
        self.step_update = 0

        layers[0].setup(self, None, layers[1])
        for i in range(1, len(layers) - 1):
            layers[i].setup(self, layers[i-1], layers[i+1])
        layers[-1].setup(self, layers[-2], None)

    def train(self, values, labels):
        if len(values) == 0:
            return

        for k in range(len(values)):
            if k % 100 == 0:
                sys.stdout.write("\r> progress: %d/%d" % (k, len(values)))

            step = self.step_init
            if self.step_update == 0:
                step /= ((k // 700) + 1)
            else:
                step *= (0.9 ** (k // 90))

            score = self.forward(values[k], labels[k])
            self.backward(score, step)
        sys.stdout.write("\r")

    def forward(self, value, label):
        for layer in self.layers:
            value = layer.forward(value, label)
        return value

    def backward(self, score, step):
        for layer in self.layers[::-1]:
            score = layer.backward(score, step)

    def predict(self, values):
        if len(values) == 0:
            return np.array([])

        predictions = np.zeros(len(values))
        for i in range(len(values)):
            if i % 100 == 0:
                sys.stdout.write("\r> progress: %d/%d" % (i, len(values)))
            output = self.forward(values[i], -1)
            predictions[i] = np.argmax(output, axis=0)
        sys.stdout.write("\r")
        return predictions

    def test(self, values, labels):
        if len(values) == 0:
            return np.array([])

        predictions = self.predict(values)
        accuracy = 0
        for i in range(len(predictions)):
            accuracy += predictions[i] == labels[i]
        return accuracy / len(predictions)


############################################################################
# Read / write data
############################################################################


def read_values(filename):
    x = []
    with open(filename) as f:
        train_reader = csv.reader(f)
        for line in train_reader:
            for index in range(len(line) - 1):
                x.append(float(line[index]))
    x = (x - np.mean(x)) / np.std(x)
    nb_inputs = int(len(x) / (3 * 32 * 32))
    x = np.reshape(x, [nb_inputs, 3, 32, 32])
    return x


def read_labels(filename):
    y = []
    with open(filename) as f:
        train_reader = csv.reader(f)
        train_reader.next()
        for line in train_reader:
            y.append(float(line[1]))
    return np.array(y)


def write_labels(filename, labels):
    with open(filename, 'w') as f:
        f.write("Id,Prediction\n")
        for i in range(len(labels)):
            f.write("%d,%d\n" % (i+1, int(labels[i])))


def print_labels(labels):
    for i in range(len(labels)):
        print("%d,%d\n" % (i+1, int(labels[i])))


############################################################################
# Main
############################################################################


def run():
    print("Building CNN...")
    cnn = Network([
        ConvLayer([3, 32, 32], 5, 2),
        ReLULayer([32, 32, 32]),
        PoolLayer([32, 32, 32]),
        ConvLayer([32, 16, 16], 5, 2),
        ReLULayer([64, 16, 16]),
        PoolLayer([64, 16, 16]),
        ConvLayer([64, 8, 8], 8, 0),
        ReLULayer([1024, 1, 1]),
        ConvLayer([1024, 1, 1], 1, 0),
        SoftLayer([10, 1, 1]),
    ])

    cnn.step_init = 0.1
    cnn.step_update = 1
    cnn.lambda_reg = 0.000
    print("- step init %f" % cnn.step_init)
    print("- step upd %d" % cnn.step_update)
    print("- step reg %f" % cnn.lambda_reg)
    print("")

    print("CNN layers:")
    for layer in cnn.layers:
        print("+ " + layer.get_desc())
    print("")

    print("Reading training data...")
    x_all = read_values('Xtr.csv')
    y_all = read_labels('Ytr.csv')

    split = int(len(y_all) * 0.90)
    print("- split %d" % split)

    x_tra = x_all[:split]
    y_tra = y_all[:split]
    x_tes = x_all[split:]
    y_tes = y_all[split:]
    print("")

    print("Training...")
    nb_epoch = 2
    for epoch in range(nb_epoch):
        permut = np.random.permutation(len(y_tra))
        (x_tmp, y_tmp) = (x_tra[permut], y_tra[permut])
        print("- training epoch: %d/%d" % (epoch, nb_epoch))
        cnn.train(x_tmp, y_tmp)
        print("- testing epoch: %d/%d" % (epoch, nb_epoch))
        accuracy = cnn.test(x_tes, y_tes)
        print("* accuracy: %f" % accuracy)
    print("")

    print("Reading predicting data...")
    x_pre = read_values('Xte.csv')
    print("")

    print("Predicting...")
    predictions = cnn.predict(x_pre)
    write_labels("Yte.csv", predictions)
    print("")

if __name__ == "__main__":
    run()

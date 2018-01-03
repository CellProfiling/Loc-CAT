"""
Module for creating and working with Deep Neural Networks.
No warranty!
"""
import itertools
import tempfile
import numpy as np
import tensorflow as tf
import iter_utils
import data_utils


# TODO: Probably move the entropy functions to another file. dnn_utils maybe?
def binary_cross_entropy(prediction, target):
    """
    Calculates the binary cross entropy value according to the below formula.
    Can only be used in tensorflow models.

    let o=prediction, t=target
    -(t*log(o) + (1-t)*log(1-o))

    Adds a small (1e-12) value to the logarithms to avoid log(0)
    """
    e = 1e-12
    op1 = tf.multiply(target, tf.log(prediction + e))
    op2 = tf.multiply(tf.subtract(1., target),
                      tf.log(tf.subtract(1., prediction) + e))
    return tf.negative(tf.add(op1, op2))


# TODO: Probably move this as well.
def gamer_cross_entropy(prediction, target, C=0.3):
    """
    Cross entropy version which includes another input as in the loss
    calculation.
    The function is aimed towards use in a tensorflow neural network.

    target should be a binary array containing two parts.
    The first half of the array should represent the true values, the "gold
    standard".
    The second half should represent the auxillary data.
    The array *has* to be evenly split by two.

    prediction should be a prediction array, at least as long as half the size
    of the target array.

    C can either be a floating point number or an array of floating point
    numbers. If it is an array, it has to be of equal length to half the size
    of the target array.

    The formula for calculation is roughly equivalent to:
        let t1 = target[0 : len(target)/2]
        let t2 = target[len(target)/2 : end(target)]
        let o = prediction[0 : len(target)/2]
        let BCE = binary_cross_entropy function
        loss = BCE(t1, o) + C * BCE(t2, o)

    Tip:
        For use in the DNN class, use functools.partial to set which C before
        building the network with the resulting function.

    Note:
        This cost function did not work out too well when we tried it out. Use
        at your own risk.
    """
    C = tf.constant(C, dtype=tf.float32)
    tensor_length = int(target.get_shape()[1])
    if tensor_length % 2:
        raise ValueError('Length of Tensor must be an even number')

    class_length = int(tensor_length/2)

    t1 = target[:, :class_length]
    t2 = target[:, class_length:]
    o = prediction[:, :class_length]

    bce1 = binary_cross_entropy(o, t1)
    bce2 = binary_cross_entropy(o, t2)

    gamer_influence = tf.multiply(C, bce2)

    return tf.add(bce1, gamer_influence)


class DNN(object):
    """
    Implements (Deep) Neural Networks using TensorFlow.
    """
    def __init__(self, model_file=None, seed=None):
        """
        Creates a DNN object.
        If no arguments are supplied, an empty network graph is created.
        Args:
            model_file: The path to a model saved previously by a call to save.
                        It is important that the meta file exists as
                        `model_file`.meta.
            seed: Sets the random seed of the tensorflow system.
        """
        if seed and model_file:
            raise ValueError('Cannot give both a model_file and a seed')

        if model_file:
            self.load(model_file)
        else:
            self._config = tf.ConfigProto(allow_soft_placement=True)
            self._graph = tf.Graph()
            self._sess = tf.Session(config=self._config, graph=self._graph)

            if seed:
                with self._graph.as_default():
                    tf.set_random_seed(seed)
                    tf.add_to_collection('seed', seed)

            self.dropouts = []
            self._drop_probs = []
            self._ops = []
            self._shapes = []
            self._activations = []

            self.built = False

    def _init_weight(self, shape, mean=0.0, stddev=0.1):
        """
        Creates a tf Variable initialized with random variable in a normal
        distribution.

        Args:
            shape:  The shape of the Variable.
            mean:   The mean of the normal distribution.
            stddev: The standard deviation of the normal distribution.
        Returns:
            An initialized tf Variable with the specified shape
        """
        with self._graph.as_default():
            distribution = tf.random_normal(shape, mean=mean, stddev=stddev)
            return tf.Variable(distribution, name='variables')

    def _init_bias(self, shape):
        """
        Creates a tf Variable initialized to be good for use as a bias
        variable.

        Args:
            shape: The shape of the bias variable
        Returns:
            An initialized bias variable.
        """
        initial = tf.random_normal(shape=shape)
        return tf.Variable(initial)

    def add_layer(self, shape, activation=tf.identity, dropout=0.0):
        """
        Adds a layer to the neural network with a specified number of output
        neurons.
        For the first layer, num_in must be specified, it is otherwise optional
        Args:
            shape:  A list containing 2 values [n_in, n_out].
            activation: The activation function used by the layer.
            dropout:    The amount of dropout to be applied to the layer,
                        specified as a value in the range [0,1].
        """

        if self.built:
            raise ValueError('Network is already built')

        self._shapes.append(list(shape))
        self._activations.append(activation)
        self._drop_probs.append(dropout)

    def cost(self, data, batch_size=500):
        """
        Calculates the cost value of the data.

        Args:
            data: A tuple containing feats and labels.
            batch_size: The number of items to process at a time.
        """
        if not self.built:
            raise ValueError('Network not built yet')

        cost = 0.
        size = 0

        feed_dict = {}
        for (dropout) in self.dropouts:
            feed_dict[dropout] = 0

        tr_data = iter_utils.iter_chunks(data[0], batch_size)
        key_values = iter_utils.iter_chunks(data[1], batch_size)
        for (tr, key) in zip(tr_data, key_values):
            feed_dict[self._X] = tr
            feed_dict[self._Y] = key
            tm = self._sess.run(self.cost_op, feed_dict=feed_dict)
            cost += tm * len(tr)
            size += len(tr)
        cost /= size

        return cost

    def build(self,
              cost=binary_cross_entropy,
              optimizer=tf.train.AdamOptimizer()):
        """
        Builds the neural network so that training and predicting can be made.
        Args:
            cost: The cost function the be used for determining the loss value
                  of the network.
                  Needs to be a function that takes two vectors as input and
                  returns a value.

            optimizer: The optimizer to run over the network.
                       This needs to be an object that has a minimize function
                       which takes a tensorflow operation and updates the
                       network based on it.
        """
        if self.built:
            raise ValueError('Network is already built')

        with self._graph.as_default():
            self._X = tf.placeholder(tf.float32, [None, self._shapes[0][0]])
            self._Y = tf.placeholder(tf.float32, [None, self._shapes[-1][1]])

            self._shapes[1][0] = self._shapes[0][0]

            # For saving and restoring model later
            tf.add_to_collection('X', self._X)
            tf.add_to_collection('Y', self._Y)

            # Special case for input layer
            h = self._activations[0](self._X)
            drop_x = tf.placeholder(tf.float32)
            drop_op = tf.nn.dropout(h, 1.-drop_x)
            tf.add_to_collection('drop0', drop_x)
            tf.add_to_collection('drop_val0', self._drop_probs[0])
            self.dropouts.append(drop_x)

            prev = drop_op
            hidden_layers = zip(self._shapes[1:], self._activations[1:])
            for i, layer in enumerate(hidden_layers, start=1):
                (shape, activation) = layer
                w_h = self._init_weight(shape)
                w_b = self._init_bias([shape[1]])

                h = activation(tf.matmul(prev, w_h) + w_b)
                drop = tf.placeholder(tf.float32)
                op_h = tf.nn.dropout(h, 1.-drop)
                self.dropouts.append(drop)
                self._ops.append(op_h)

                # For saving and restoring model later
                tf.add_to_collection('drop%d' % i, drop)
                tf.add_to_collection('drop_val%d' % i, self._drop_probs[i])

                prev = op_h

            self.cost_op = tf.reduce_mean(cost(prev, self._Y))
            self.train_op = optimizer.minimize(self.cost_op)
            self.predict_op = prev

            # For saving and restoring model later
            tf.add_to_collection('cost', self.cost_op)
            tf.add_to_collection('train', self.train_op)
            tf.add_to_collection('predict', self.predict_op)

            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
            self.built = True

    def train(self, train, validation_data=None, epochs=100, batch_size=500,
              verbose=False, early_stopping=None):
        """
        Trains the network on the input data.
        Args:
            train: A tuple or list on the format [input, output] where input is
                   a list  of training data and output is a list of the correct
                   labels for the corresponding data.

                   Implementation specific detail:
                   generators can also be used as long as they are able to be
                   repeated. The data is accessed by calling (x0, x1) = train
                   each epoch, which can be utilized if neccessary.

            validation_data: Validation data, which is used to validate the
                             network each epoch.

            epochs: The number of epochs to train the network.

            batch_size: The number of items to use as network input at a time.
                        Most relevant when working on a GPU to determine the
                        number of data points to transfer to the GPU at a time.

            verbose: Determines if cost should be printed for each epoch.

            early_stopping: Determines if early stopping should be used.
                            If None, no early stopping will be used.
                            If it is an integer, training will stop after
                            `early_stopping` epochs of non-decreasing
                            validation cost.

                            Does not have an effect if there is no validation
                            set.
        """
        validation_string = ('\t  Validation acc(cutoff 0.4): {:.6f}\n'
                             '\t  Validation cost: {:.6f}')
        if not self.built:
            raise ValueError('Network not built yet')

        cost = 0

        tmp_storage_file = tempfile.NamedTemporaryFile()
        with self._graph.as_default():
            feed_dict = {}
            dropouts = zip(self.dropouts, self._drop_probs)
            for (dropout, p_drop) in dropouts:
                feed_dict[dropout] = p_drop

            previous_val_cost = 999999999
            non_decreasing_epochs = 0
            for i in range(epochs):

                (train_data, train_keys) = train
                tr_data = iter_utils.iter_chunks(train_data, batch_size)
                key_values = iter_utils.iter_chunks(train_keys, batch_size)
                for (tr, key) in zip(tr_data, key_values):
                    feed_dict[self._X] = tr
                    feed_dict[self._Y] = key
                    self._sess.run(self.train_op, feed_dict=feed_dict)

                if verbose:
                    cost = self.cost(train, batch_size)
                    predictions = np.asarray(self.predict(train_data))
                    predictions[predictions > 0.4] = 1
                    predictions[predictions < 1.0] = 0
                    acc = data_utils.hamming_score(train_keys, predictions)
                    formatstring = '{} cost: {:.6f}  acc(cutoff 0.4): {:.6f}'
                    istr = str(i).zfill(len(str(epochs)))
                    print(formatstring.format(istr, cost, acc))

                if validation_data:
                    val_cost = self.cost(validation_data, batch_size)
                    if val_cost < previous_val_cost:
                        previous_val_cost = val_cost
                        non_decreasing_epochs = 0
                        if early_stopping:
                            self.save(tmp_storage_file.name)
                    else:
                        non_decreasing_epochs += 1
                    predictions = np.asarray(self.predict(validation_data[0]))
                    predictions[predictions > 0.4] = 1
                    predictions[predictions < 1.0] = 0
                    acc = data_utils.hamming_score(validation_data[1],
                                                   predictions)
                    if verbose:
                        print(validation_string.format(acc, val_cost), end=' ')
                        if early_stopping:
                            epoch_left = early_stopping - non_decreasing_epochs
                            print('(Patience:', epoch_left, ')')
                        else:
                            print()

                    if not early_stopping:
                        continue
                    # Break out of training if we don't get improved
                    # performance on validation set
                    if (non_decreasing_epochs >= early_stopping):
                        self.load(tmp_storage_file.name)
                        break
        return cost

    def predict(self, data, batch_size=500):
        """
        Predicts on input data.
        Args:
            data:   A list of data on the same format that the network was
                    trained on.
            batch_size: How many items to predict on at a time.
        Returns:
            A list of predictions based on the internal classification model.
        """
        if not self.built:
            raise ValueError('Network not built yet')
        with self._graph.as_default():
            feed_dict = {}
            for dropout in self.dropouts:
                feed_dict[dropout] = 0.

            predictions = []
            for tr in iter_utils.iter_chunks(data, batch_size):
                feed_dict[self._X] = tr
                tmp = self._sess.run(self.predict_op, feed_dict=feed_dict)
                predictions.extend(tmp)
            return predictions

    def save(self, path):
        """
        Saves the neural network model to file called `path`.
        Writes part of the model into a separate meta file, named `path`.meta.
        """
        with self._graph.as_default():
            saver = tf.train.Saver(sharded=False)
            saver.save(self._sess, path, write_meta_graph=True)

    def load(self, model_file):
        """
        Zeroes the current model and loads a previously trained one from file.
        It is important that both the file `model_file` and `model_file.meta`
        are available when loading the data.

        Args:
            model_file: The path from which to load the saved model.
                        Will also use the file model_file.meta
        """
        self._config = tf.ConfigProto(allow_soft_placement=True)
        self._graph = tf.Graph()
        self._sess = tf.Session(config=self._config, graph=self._graph)
        self.dropouts = []
        self._drop_probs = []
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(model_file + '.meta')
            saver.restore(self._sess, model_file)

            if len(tf.get_collection('seed')):
                tf.set_random_seed(tf.get_collection('seed')[0])

            self._X = tf.get_collection('X')[0]
            self._Y = tf.get_collection('Y')[0]
            self.predict_op = tf.get_collection('predict')[0]
            self.cost_op = tf.get_collection('cost')[0]
            self.train_op = tf.get_collection('train')[0]
            for i in itertools.count():
                dropout = tf.get_collection('drop%d' % i)
                val = tf.get_collection('drop_val%d' % i)
                if len(val):
                    self.dropouts.append(dropout[0])
                    self._drop_probs.append(val[0])
                else:
                    break
        self.built = True

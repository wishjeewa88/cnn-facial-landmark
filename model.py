"""This module provides the function to build the network."""
import tensorflow as tf
from tensorflow import keras


def build_landmark_model(input_shape, output_size):
    """Build the convolutional network model with Keras Functional API.

    Args:
        input_shape: the shape of the input image, without batch size.
        output_size: the number of output node, usually equals to the number of
            marks times 2 (in 2d space).

    Returns:
        a Keras model, not compiled.
    """

    # The model is composed of multiple layers.

    # Preprocessing layers.
    preprocess = keras.layers.experimental.preprocessing.Normalization()

    # Convolutional layers.
    conv_1 = keras.layers.Conv2D(filters=32,
                                 kernel_size=(3, 3),
                                 activation='relu')
    conv_2 = keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu')
    conv_3 = keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu')
    conv_4 = keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu')
    conv_5 = keras.layers.Conv2D(filters=64,
                                 kernel_size=[3, 3],
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu')
    conv_6 = keras.layers.Conv2D(filters=128,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu')
    conv_7 = keras.layers.Conv2D(filters=128,
                                 kernel_size=[3, 3],
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu')
    conv_8 = keras.layers.Conv2D(filters=256,
                                 kernel_size=[3, 3],
                                 strides=(1, 1),
                                 padding='valid',
                                 activation='relu')

    # Pooling layers.
    pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='valid')
    pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='valid')
    pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='valid')
    pool_4 = keras.layers.MaxPool2D(pool_size=[2, 2],
                                    strides=(1, 1),
                                    padding='valid')

    # Dense layers.
    #dense_1 = keras.layers.Dense(units=1024,
                                 #activation='relu',
                                 #use_bias=True)
    dense_1 = keras.layers.Dense(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
    dense_2 = keras.layers.Dense(units=output_size,
                                 activation=None,
                                 use_bias=True)
    
    class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
    
    
    # Batch norm layers
    bn_1 = keras.layers.BatchNormalization()
    bn_2 = keras.layers.BatchNormalization()
    bn_3 = keras.layers.BatchNormalization()
    bn_4 = keras.layers.BatchNormalization()
    bn_5 = keras.layers.BatchNormalization()
    bn_6 = keras.layers.BatchNormalization()
    bn_7 = keras.layers.BatchNormalization()
    bn_8 = keras.layers.BatchNormalization()
    bn_9 = keras.layers.BatchNormalization()


    # Flatten layers.
    flatten_1 = keras.layers.Flatten()

    # All layers got. Define the forward propgation.
    inputs = keras.Input(shape=input_shape, name="image_input")

    # Preprocess the inputs.
    x = preprocess(inputs)

    # |== Layer 1 ==|
    x = conv_1(x)
    x = bn_1(x)
    x = pool_1(x)

    # |== Layer 2 ==|
    x = conv_2(x)
    x = bn_2(x)
    x = conv_3(x)
    x = bn_3(x)
    x = pool_2(x)

    # |== Layer 3 ==|
    x = conv_4(x)
    x = bn_4(x)
    x = conv_5(x)
    x = bn_5(x)
    x = pool_3(x)

    # |== Layer 4 ==|
    x = conv_6(x)
    x = bn_6(x)
    x = conv_7(x)
    x = bn_7(x)
    x = pool_4(x)

    # |== Layer 5 ==|
    x = conv_8(x)
    x = bn_8(x)

    # |== Layer 6 ==|
    x = flatten_1(x)
    x = dense_1(x)
    x = bn_9(x)
    outputs = dense_2(x)

    # Return the model
    return keras.Model(inputs=inputs, outputs=outputs, name="landmark")

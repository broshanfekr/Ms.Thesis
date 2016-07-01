import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np


class LSTM_class(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, embedding_size, n_classes, n_hidden, vocab_size):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_y')
        self.istate = tf.placeholder("float", [None, 2 * n_hidden], name='istate')  # state & cell => 2x n_hidden

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # n_input = embedding_size
        # s_steps = sequence_length

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_embedding')

            self.embedded_chars = tf.nn.embedding_lookup(W_embedding, self.input_x)
            self.embedded_chars = tf.transpose(self.embedded_chars, [1, 0, 2], name='transpose_input')
            # Reshape to prepare input to hidden activation
            self.embedded_chars = tf.reshape(self.embedded_chars, [-1, embedding_size], name='reshape_imput')

        with tf.name_scope("mapping_to_hidden"):
            self.W_hidden = tf.Variable(tf.random_normal([embedding_size, n_hidden]),
                                        name='W_hidden')  # Hidden layer weights
            self.B_hidden = tf.Variable(tf.random_normal([n_hidden]), name='B_hidden')

            # Linear activation
            self.hidden_matrix = tf.add(tf.matmul(self.embedded_chars, self.W_hidden), self.B_hidden)
            self.hidden_matrix = tf.split(0, sequence_length, self.hidden_matrix, name="hidden_matrix")

            self.outputs, self.states = rnn.rnn(rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0), self.hidden_matrix
                                                , initial_state=self.istate)

            self.sum_outputs = tf.add_n(self.outputs)
            self.normal_output = tf.nn.l2_normalize(self.sum_outputs, dim=0)

        with tf.name_scope("output"):
            self.W_out = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='W_out')
            self.B_out = tf.Variable(tf.random_normal([n_classes]), name='B_out')
            self.output_layer = tf.add(tf.matmul(self.normal_output, self.W_out), self.B_out, 'output_layer')
            self.predictions = tf.argmax(self.output_layer, 1, name='predictions')

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.l2_loss = tf.nn.l2_loss(self.W_hidden) + tf.nn.l2_loss(self.W_out) \
                           + tf.nn.l2_loss(self.B_hidden) + tf.nn.l2_loss(self.B_out, name='l2_loss')
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.output_layer, self.input_y)) + l2_loss  # Softmax loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1), name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name='accuracy')

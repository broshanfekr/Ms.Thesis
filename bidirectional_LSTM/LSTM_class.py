import tensorflow as tf
import numpy as np

def dynamic_bidirectional_LSTM(input_x, seq_len_list, n_hidden, weights, biases,dropout_keep_prob):

    with tf.name_scope("mapping_to_hidden"):

        hidden_matrix = input_x

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=dropout_keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,
                                                          cell_bw=lstm_cell,
                                                          inputs=hidden_matrix,
                                                          dtype=tf.float32,
                                                          sequence_length=seq_len_list)

        output_fw, output_bw = outputs
        states_fw, states_bw = states

        outputs = tf.concat([output_fw, output_bw], 2)

        seq_len_list = tf.reshape(seq_len_list, [-1, 1])
        seq_len_tmp = tf.cast(seq_len_list, tf.float32)

        sum_outputs = tf.reduce_sum(outputs, reduction_indices=1)
        normal_output = tf.div(sum_outputs, seq_len_tmp)



    with tf.name_scope("output"):
        output_layer = tf.add(tf.matmul(normal_output, weights['W_out']), biases['B_out'], 'output_layer')

    return output_layer , [outputs, normal_output]
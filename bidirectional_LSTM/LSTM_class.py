import tensorflow as tf
import numpy as np

def dynamic_bidirectional_LSTM(input_x, embedding_size, seq_max_len, seq_len_list, n_hidden, weights, biases,dropout_keep_prob):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, sequence_length, embedding_size)
    # Required shape: 'sequence_length' tensors list of shape (embedding_size, n_input)
    # Permuting batch_size and sequence_length
    #with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #W_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_embedding')
        #self.embedded_chars = tf.nn.embedding_lookup(W_embedding, self.input_x)
        #embedded_chars = tf.transpose(input_x, [1, 0, 2], name='transpose_input')
    #embedded_chars = tf.reshape(input_x, [-1, embedding_size], name='reshape_imput')

    with tf.name_scope("mapping_to_hidden"):

        #hidden_matrix = tf.add(tf.matmul(embedded_chars, weights['W_hidden']), biases['B_hidden'])
        #hidden_matrix = tf.reshape(hidden_matrix, [-1, seq_max_len, n_hidden])
        #hidden_matrix = tf.split(0, seq_max_len, hidden_matrix, name="hidden_matrix")
        hidden_matrix = input_x

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=dropout_keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,
                                                          cell_bw=lstm_cell,
                                                          inputs=hidden_matrix,
                                                          dtype=tf.float32,
                                                          sequence_length=seq_len_list)

        output_fw, output_bw = outputs
        states_fw, states_bw = states

        outputs = tf.concat(2 ,[output_fw, output_bw])

        seq_len_list = tf.reshape(seq_len_list, [-1, 1])
        seq_len_tmp = tf.cast(seq_len_list, tf.float32)

        sum_outputs = tf.reduce_sum(outputs, reduction_indices=1)
        normal_output = tf.div(sum_outputs, seq_len_tmp)
        #normal_output = tf.nn.l2_normalize(sum_outputs, dim=1)

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, sequence_length, embedding_size]
        #batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        #index = tf.range(0, batch_size) * seq_max_len + (tf.to_int32(seq_len_list) - 1)
        # Indexing
        #normal_output = tf.gather(tf.reshape(outputs, [-1, 2*n_hidden]), index)
        #mean_outputs = tf.reduce_mean(outputs, reduction_indices=1)
        #sum_outputs = tf.reduce_sum(outputs, reduction_indices=1)
        #normal_output = tf.nn.l2_normalize(mean_outputs, dim=0)



    with tf.name_scope("output"):
        output_layer = tf.add(tf.matmul(normal_output, weights['W_out']), biases['B_out'], 'output_layer')

    return output_layer , [outputs, normal_output]
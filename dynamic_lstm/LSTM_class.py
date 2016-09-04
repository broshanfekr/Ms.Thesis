import tensorflow as tf
import numpy as np


def LSTM(input_x, embedding_size, seq_max_len, seq_len_list, n_hidden, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, sequence_length, embedding_size)
    # Required shape: 'sequence_length' tensors list of shape (embedding_size, n_input)
    # Permuting batch_size and sequence_length
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #W_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_embedding')
        #self.embedded_chars = tf.nn.embedding_lookup(W_embedding, self.input_x)
        embedded_chars = tf.transpose(input_x, [1, 0, 2], name='transpose_input')
        embedded_chars = tf.reshape(embedded_chars, [-1, embedding_size], name='reshape_imput')

    with tf.name_scope("mapping_to_hidden"):

        hidden_matrix = tf.add(tf.matmul(embedded_chars, weights['W_hidden']), biases['B_hidden'])
        hidden_matrix = tf.split(0, seq_max_len, hidden_matrix, name="hidden_matrix")

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
        #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_keep_prob)

        outputs, states = tf.nn.rnn(cell=lstm_cell, inputs=hidden_matrix,
                                    dtype=tf.float32, sequence_length=seq_len_list)

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, sequence_length, embedding_size]
        '''
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seq_len_list - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
        normal_output = outputs
        '''
        sum_outputs = tf.add_n(outputs)
        normal_output = tf.nn.l2_normalize(sum_outputs, dim=0)

    with tf.name_scope("output"):
        output_layer = tf.add(tf.matmul(normal_output, weights['W_out']), biases['B_out'], 'output_layer')

    return output_layer

def dynamic_LSTM(input_x, embedding_size, seq_max_len, seq_len_list, n_hidden, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, sequence_length, embedding_size)
    # Required shape: 'sequence_length' tensors list of shape (embedding_size, n_input)
    # Permuting batch_size and sequence_length
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #W_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_embedding')
        #self.embedded_chars = tf.nn.embedding_lookup(W_embedding, self.input_x)
        #embedded_chars = tf.transpose(input_x, [1, 0, 2], name='transpose_input')
        embedded_chars = tf.reshape(input_x, [-1, embedding_size], name='reshape_imput')

    with tf.name_scope("mapping_to_hidden"):

        hidden_matrix = tf.add(tf.matmul(embedded_chars, weights['W_hidden']), biases['B_hidden'])
        hidden_matrix = tf.reshape(hidden_matrix, [-1, seq_max_len, n_hidden])
        #hidden_matrix = tf.split(0, seq_max_len, hidden_matrix, name="hidden_matrix")

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
        #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_keep_prob)

        outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=hidden_matrix,
                                            dtype=tf.float32, sequence_length=seq_len_list)

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, sequence_length, embedding_size]
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seq_len_list - 1)
        # Indexing
        normal_output = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        #mean_outputs = tf.reduce_mean(outputs, reduction_indices=1)
        #sum_outputs = tf.reduce_sum(outputs, reduction_indices=1)
        #normal_output = tf.nn.l2_normalize(mean_outputs, dim=0)


    with tf.name_scope("output"):
        output_layer = tf.add(tf.matmul(normal_output, weights['W_out']), biases['B_out'], 'output_layer')

    return output_layer

def dynamic_bidirectional_LSTM(input_x, embedding_size, seq_max_len, seq_len_list, n_hidden, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, sequence_length, embedding_size)
    # Required shape: 'sequence_length' tensors list of shape (embedding_size, n_input)
    # Permuting batch_size and sequence_length
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #W_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W_embedding')
        #self.embedded_chars = tf.nn.embedding_lookup(W_embedding, self.input_x)
        #embedded_chars = tf.transpose(input_x, [1, 0, 2], name='transpose_input')
        embedded_chars = tf.reshape(input_x, [-1, embedding_size], name='reshape_imput')

    with tf.name_scope("mapping_to_hidden"):

        hidden_matrix = tf.add(tf.matmul(embedded_chars, weights['W_hidden']), biases['B_hidden'])
        hidden_matrix = tf.reshape(hidden_matrix, [-1, seq_max_len, n_hidden])
        #hidden_matrix = tf.split(0, seq_max_len, hidden_matrix, name="hidden_matrix")

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
        #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.dropout_keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,
                                                          cell_bw=lstm_cell,
                                                          inputs=hidden_matrix,
                                                          dtype=tf.float32,
                                                          sequence_length=seq_len_list)

        output_fw, output_bw = outputs
        states_fw, states_bw = states

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, sequence_length, embedding_size]
        '''
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seq_len_list - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
        normal_output = outputs
        '''
        outputs = tf.add(output_fw, output_bw)
        mean_outputs = tf.reduce_mean(outputs, reduction_indices=1)
        #sum_outputs = tf.reduce_sum(outputs, reduction_indices=1)
        normal_output = tf.nn.l2_normalize(mean_outputs, dim=0)

    '''
    with tf.name_scope("attention"):
        self.W_hidden_attention = tf.Variable(tf.random_normal([n_hidden, n_hidden_attention]),
                                              name="W_hidden_attention")
        self.b_hidden_attention = tf.Variable(tf.random_normal([n_hidden_attention]),
                                              name = "b_hidden_attention")

        self.attention_weigth = tf.Variable(tf.random_normal([n_hidden_attention,1]),
                                            name="attention_weigth")

        tmp_outputs = tf.reshape(self.outputs, [-1, n_hidden])

        Ut = tf.tanh(tf.add(tf.matmul(tmp_outputs, self.W_hidden_attention), self.b_hidden_attention))
        alfa = tf.matmul(Ut, self.attention_weigth)
        alfa = tf.reshape(alfa, [-1, sequence_length])
        alfa = tf.nn.softmax(alfa)
        alfa = tf.reshape(alfa, [-1, 1])
        tmp = tf.ones([1, n_hidden])
        alfa = tf.matmul(alfa, tmp)
        alfa = tf.reshape(alfa, [-1, sequence_length, n_hidden])
        self.outputs = tf.mul(self.outputs, alfa)
        self.outputs = tf.transpose(self.outputs, [1, 0, 2])
        self.outputs = tf.reshape(self.outputs, [-1, n_hidden])
        self.outputs = tf.split(0, sequence_length, self.outputs)
    '''

    with tf.name_scope("output"):
        output_layer = tf.add(tf.matmul(normal_output, weights['W_out']), biases['B_out'], 'output_layer')

    return output_layer
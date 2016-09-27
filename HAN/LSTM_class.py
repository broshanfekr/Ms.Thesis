import tensorflow as tf
import numpy as np


def dynamic_bidirectional_LSTM(input_x, embedding_size, seq_max_len, doc_len_list, sent_len_list,
                               n_hidden, weights, biases, sent_mask_list, doc_mask_list, dropout_keep_prob):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, sequence_length, embedding_size)
    # Required shape: 'sequence_length' tensors list of shape (embedding_size, n_input)
    # Permuting batch_size and sequence_length
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedded_words = tf.reshape(input_x, [-1, embedding_size], name='reshape_input')

    with tf.variable_scope("word_LSTM"):
        hidden_matrix = tf.add(tf.matmul(embedded_words, weights['W_hidden']), biases['B_hidden'])
        #hidden_matrix = tf.reshape(hidden_matrix, [-1, seq_max_len[0], seq_max_len[1], n_hidden])

        hidden_matrix = tf.reshape(hidden_matrix, [-1, seq_max_len[1], n_hidden])

        lstm_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=dropout_keep_prob)
        sent_output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,
                                                              cell_bw=lstm_cell,
                                                              inputs=hidden_matrix,
                                                              dtype=tf.float32,
                                                              sequence_length=sent_len_list)

        output_fw, output_bw = sent_output
        states_fw, states_bw = states

        sent_output = tf.concat(2 ,[output_fw, output_bw])

        #computing weights for each word in the sentence
        tmp_outputs = tf.reshape(sent_output, [-1, 2*n_hidden])
        Ut = tf.tanh(tf.add(tf.matmul(tmp_outputs, weights['W_word_attention']), biases['b_word_attention']))

        first_alfa = tf.exp(tf.matmul(Ut, weights['word_attention_vector']))

        sent_alfa = tf.reshape(first_alfa, [-1, seq_max_len[1]])
        sent_alfa = tf.mul(sent_alfa, sent_mask_list)

        Normalize_factor = tf.reduce_sum(sent_alfa, 1, keep_dims=True)
        Normalize_factor = tf.add(Normalize_factor, 1.0e-20)
        sent_alfa = tf.div(sent_alfa, Normalize_factor)
        tmp_weight = tf.reduce_sum(sent_alfa, 1)

        sent_alfa = tf.reshape(sent_alfa, [-1, seq_max_len[1], 1])
        outputs = tf.mul(sent_output, sent_alfa)

        weigthed_sent_outputs = tf.reshape(outputs, [-1, seq_max_len[0], seq_max_len[1], 2*n_hidden])
        sent_lstm_outputs = tf.reduce_sum(weigthed_sent_outputs, reduction_indices=2)


    with tf.variable_scope("sent_LSTM"):
        # Define a lstm cell with tensorflow
        sent_lstm_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        #sent_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=sent_lstm_cell, output_keep_prob=dropout_keep_prob)
        doc_output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=sent_lstm_cell, cell_bw=sent_lstm_cell,
                                                             inputs=sent_lstm_outputs,
                                                             dtype=tf.float32,
                                                             sequence_length=doc_len_list)

        output_fw, output_bw = doc_output
        states_fw, states_bw = states

        doc_output = tf.concat(2, [output_fw, output_bw])

        # computing weights for each word in the sentence
        tmp_outputs = tf.reshape(doc_output, [-1, 2*n_hidden])
        Ut = tf.tanh(tf.add(tf.matmul(tmp_outputs, weights['W_sent_attention']), biases['b_sent_attention']))

        doc_alfa = tf.exp(tf.matmul(Ut, weights['sent_attention_vector']))

        doc_alfa = tf.reshape(doc_alfa, [-1, seq_max_len[0]])
        doc_alfa = tf.mul(doc_alfa, doc_mask_list)

        Normalize_factor = tf.reduce_sum(doc_alfa, 1, keep_dims=True)
        Normalize_factor = tf.add(Normalize_factor, 1.0e-20)
        doc_alfa = tf.div(doc_alfa, Normalize_factor)
        tmp2_weight = tf.reduce_sum(doc_alfa, 1)

        doc_alfa = tf.reshape(doc_alfa, [-1, seq_max_len[0], 1])
        outputs = tf.mul(doc_output, doc_alfa)

        final_output = tf.reduce_sum(outputs, reduction_indices=1)



    with tf.name_scope("output"):
        output_layer = tf.add(tf.matmul(final_output, weights['W_out']), biases['B_out'], 'output_layer')

    return output_layer#, [embedded_words,
                       #   hidden_matrix,
                       #   sent_output,
                       #   first_alfa,
                       #   sent_alfa,
                       #   tmp_weight,
                       #   weigthed_sent_outputs,
                       #   sent_lstm_outputs,
                       #   doc_output,
                       #   doc_alfa,
                       #   tmp2_weight,
                       #   final_output,
                       #   output_layer]



'''
hidden_matrix = tf.transpose(hidden_matrix, [1, 0, 2, 3])
hidden_matrix = tf.unpack(hidden_matrix,axis=0, name="hidden_matrix")
outputs = tf.constant([], dtype=tf.float32)
index = tf.constant(0)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)

def body(x,outputs_p,i):
    myinput = tf.gather_nd(x, [[i]], name=None)
    myinput = tf.reshape(myinput, [-1, seq_max_len[1], n_hidden])
    outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=myinput,
                                        dtype=tf.float32, sequence_length=sent_len_list)


    tmp_outputs = tf.reshape(outputs, [-1, n_hidden])

    Ut = tf.tanh(tf.add(tf.matmul(tmp_outputs, weights['W_word_attention']), biases['b_word_attention']))
    alfa = tf.matmul(Ut, weights['word_attention_weigth'])
    alfa = tf.reshape(alfa, [-1, seq_max_len[1]])
    alfa = tf.nn.softmax(alfa)
    alfa = tf.reshape(alfa, [-1, 1])
    tmp = tf.ones([1, n_hidden])
    alfa = tf.matmul(alfa, tmp)
    alfa = tf.reshape(alfa, [-1, seq_max_len[1], n_hidden])
    outputs = tf.mul(outputs, alfa)
    #mean_outputs = tf.reduce_mean(outputs, reduction_indices=1)
    outputs = tf.reduce_sum(outputs, reduction_indices=1)
    #outputs = tf.nn.l2_normalize(sum_outputs, dim=0)

    #outputs = tf.transpose(outputs, [1, 0, 2])
    #outputs = tf.reshape(outputs, [-1, n_hidden])
    #outputs = tf.split(0, seq_max_len[1], outputs)

    outputs = tf.reshape(outputs, [-1])
    outputs_n = tf.concat(0, [outputs_p, outputs])
    i = tf.add(i, 1)
    return x, outputs_n, i

def condition(x,outputs,i):
    return i < 10
hidden_matrix, outputs, index = tf.while_loop(condition, body, [hidden_matrix, outputs, index])
'''

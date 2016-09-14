import tensorflow as tf
import numpy as np

def dynamic_bidirectional_LSTM(input_x, embedding_size, seq_max_len, doc_len_list, sent_len_list,
                               n_hidden, weights, biases, sent_mask_list, doc_mask_list, dropout_keep_prob):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, sequence_length, embedding_size)
    # Required shape: 'sequence_length' tensors list of shape (embedding_size, n_input)
    # Permuting batch_size and sequence_length
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedded_words = tf.reshape(input_x, [-1, embedding_size], name='reshape_imput')

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

        tmp_outputs = tf.reshape(sent_output, [-1, 2*n_hidden])

        Ut = tf.tanh(tf.add(tf.matmul(tmp_outputs, weights['W_word_attention']), biases['b_word_attention']))
        first_alfa = tf.matmul(Ut, weights['word_attention_vector'])
        alfa = tf.reshape(first_alfa, [-1, seq_max_len[1]])
        sent_word_weigth = tf.nn.softmax(alfa)

        sent_word_weigth = tf.mul(sent_word_weigth, sent_mask_list)
        sent_word_weigth = tf.nn.l2_normalize(sent_word_weigth, dim=1)

        alfa = tf.reshape(sent_word_weigth, [-1, 1])
        tmp = tf.ones([1, 2*n_hidden])
        weigthed_words = tf.matmul(alfa, tmp)
        weigthed_words = tf.reshape(weigthed_words, [-1, seq_max_len[1], 2*n_hidden])
        outputs = tf.mul(sent_output, weigthed_words)
        weigthed_sent_outputs = tf.reshape(outputs, [-1, seq_max_len[0], seq_max_len[1], 2*n_hidden])
        sent_lstm_outputs = tf.reduce_sum(weigthed_sent_outputs, reduction_indices=2)

    with tf.variable_scope("sent_LSTM"):
        # Define a lstm cell with tensorflow
        sent_lstm_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        doc_output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=sent_lstm_cell,
                                                             cell_bw=sent_lstm_cell,
                                                             inputs=sent_lstm_outputs,
                                                             dtype=tf.float32,
                                                             sequence_length=doc_len_list)

        output_fw, output_bw = doc_output
        states_fw, states_bw = states

        doc_output = tf.concat(2, [output_fw, output_bw])

        tmp_outputs = tf.reshape(doc_output, [-1, 2*n_hidden])

        Ut = tf.tanh(tf.add(tf.matmul(tmp_outputs, weights['W_sent_attention']), biases['b_sent_attention']))
        sec_alfa = tf.matmul(Ut, weights['sent_attention_vector'])
        alfa = tf.reshape(sec_alfa, [-1, seq_max_len[0]])
        doc_weigth = tf.nn.softmax(alfa)

        doc_weigth = tf.mul(doc_weigth, doc_mask_list)
        doc_weigth = tf.nn.l2_normalize(doc_weigth, dim=1)
        doc_weigth = tf.nn.l2_normalize(doc_weigth, dim=1)
        doc_weigth = tf.nn.l2_normalize(doc_weigth, dim=1)
        doc_weigth = tf.nn.l2_normalize(doc_weigth, dim=1)

        alfa = tf.reshape(doc_weigth, [-1, 1])
        tmp = tf.ones([1, 2*n_hidden])
        weigthed_docs = tf.matmul(alfa, tmp)
        weigthed_docs = tf.reshape(weigthed_docs, [-1, seq_max_len[0], 2*n_hidden])
        outputs = tf.mul(doc_output, weigthed_docs)
        final_output = tf.reduce_sum(outputs, reduction_indices=1)

    with tf.name_scope("output"):
        output_layer = tf.add(tf.matmul(final_output, weights['W_out']), biases['B_out'], 'output_layer')

    return output_layer#, [embedded_words,
                       #   hidden_matrix,
                       #   sent_output,
                       #   first_alfa,
                       #   sent_word_weigth,
                       #   weigthed_words,
                       #   weigthed_sent_outputs,
                       #   sent_lstm_outputs,
                       #   doc_output,
                       #   doc_weigth,
                       #   weigthed_docs,
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

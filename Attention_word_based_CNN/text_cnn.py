import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
        self, sequence_length, num_classes,
        embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, n_hidden_attention=50):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.istrue = tf.placeholder(tf.bool, name="is_true")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        attention_l2_loss = tf.constant(0.0)

        # Embedding layer

        self.embedded_words_expanded = tf.expand_dims(self.input_x, -1)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        self.myoutputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer

                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W", trainable=True)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b", trainable=True)

                self.filter_W = W

                conv = tf.nn.conv2d(
                    self.embedded_words_expanded,
                    W,
                    #strides=[1, 1, 150, 1],#if padding=="SAME" was used
                    strides=[1, 1, 1, 1], # if padding == VALID was used
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.myoutputs.append(h)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    #ksize=[1, sequence_length, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                filter_seq_len = sequence_length - filter_size + 1
                tmp_scale = tf.constant(float(1.0/filter_seq_len))
                #pooled = tf.mul(pooled, tmp_scale)


                pooled_outputs.append(pooled)


            # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)


        #usual cnn uses these two lines
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_pool_flat = tf.tanh(self.h_pool_flat)
        #self.h_pool_flat = tf.nn.l2_normalize(self.h_pool_flat, dim=1)




        ##############################################################################################################
        self.W_word_attention = tf.Variable(tf.random_normal([num_filters, n_hidden_attention]), name="W_word_attention", trainable=True)
        self.b_word_attention = tf.Variable(tf.random_normal([n_hidden_attention]), name="b_word_attention", trainable=True)
        self.attention_vector = tf.Variable(tf.zeros([n_hidden_attention, 1]), name="attention_vector", trainable=True)

        last_layer_W = tf.get_variable(
            "last_layer_W",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        last_layer_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="last_layer_b", trainable=True)

        l2_loss += tf.nn.l2_loss(last_layer_W)
        l2_loss += tf.nn.l2_loss(last_layer_b)
        attention_l2_loss += tf.nn.l2_loss(last_layer_W)
        attention_l2_loss += tf.nn.l2_loss(last_layer_b)
        ###############################################################################################################
        self.myatt = self.attention_vector

        #attention_l2_loss += tf.nn.l2_loss(W_word_attention)
        #attention_l2_loss += tf.nn.l2_loss(b_word_attention)
        #attention_l2_loss += tf.nn.l2_loss(attention_vector)
        ##############################################################################################################

        filter_features = []
        for i, filter_size in enumerate(filter_sizes):

            filter_seq_len = sequence_length - filter_size + 1 # sequence length after applying this filter


            filter_outputs = self.myoutputs[i]
            attention_h_pool = tf.transpose(filter_outputs, [0, 2, 1, 3], name="transpose_op")
            attention_h_pool = tf.reshape(attention_h_pool, [-1, filter_seq_len, num_filters])

            with tf.name_scope("attention-%s" % filter_sizes[i]):

                tmp_h_pool = tf.reshape(attention_h_pool, [-1, num_filters])
                Ut = tf.tanh(tf.add(tf.matmul(tmp_h_pool, self.W_word_attention), self.b_word_attention))

                #doc_alfa = tf.exp(tf.matmul(Ut, attention_vector))
                doc_alfa = tf.matmul(Ut, self.attention_vector)

                doc_alfa = tf.reshape(doc_alfa, [-1, filter_seq_len])
                doc_alfa = tf.nn.softmax(doc_alfa)
                #Normalize_factor = tf.reduce_sum(doc_alfa, 1, keep_dims=True)
                #Normalize_factor = tf.add(Normalize_factor, 1.0e-15)
                #doc_alfa = tf.div(doc_alfa, Normalize_factor)
                #self.Normalize_factor_sum = tf.reduce_sum(doc_alfa, 1)  # to check if normalization is correct.

                doc_alfa = tf.reshape(doc_alfa, [-1, filter_seq_len, 1])

                doc_alfa = tf.scalar_mul(filter_seq_len, doc_alfa)

                #tmp_scale = tf.constant(float(1.0 / filter_seq_len))
                self.doc_alfa = doc_alfa
                outputs = tf.multiply(attention_h_pool, doc_alfa)           #+++++++++++++++++++++++++++++++++++


                attention_h_pool_flat = tf.reduce_max(outputs, reduction_indices=1)
                filter_features.append(attention_h_pool_flat)




        # attention cnn uses these three line
        self.attention_h_pool_flat = tf.concat(filter_features, 1)
        self.attention_h_pool_flat = tf.tanh(self.attention_h_pool_flat)
        #self.attention_h_pool_flat = tf.nn.l2_normalize(self.attention_h_pool_flat, dim=1)

        # self.attention_h_pool = tf.transpose(self.attention_h_pool, [0,2,1,3], name="transpose_op")
        # self.attention_h_pool = tf.reshape(self.attention_h_pool, [-1, sequence_length, num_filters_total])

        #tmp = np.asarray([[3.0, 5.0, 7.0], [7.0, 5.0, 3.0]])
        #tmp = tf.constant(tmp)
        #self.myattention_output = tf.nn.l2_normalize(tmp, dim=1)


##########################################################################################################################################
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.scores = tf.nn.xw_plus_b(self.h_drop, last_layer_W, last_layer_b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.weigth_norm = l2_reg_lambda*l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.correct_predictions = correct_predictions
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

##########################################################################################################################################

        # Add dropout
        with tf.name_scope("attention_dropout"):
            self.attention_h_drop = tf.nn.dropout(self.attention_h_pool_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("attention_output"):
            self.attention_scores = tf.nn.xw_plus_b(self.attention_h_drop, last_layer_W, last_layer_b, name="scores")
            self.attention_predictions = tf.argmax(self.attention_scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("attention_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.attention_scores, labels=self.input_y)
            self.attention_loss = tf.reduce_mean(losses) + l2_reg_lambda * attention_l2_loss
            self.attention_weigth_norm = l2_reg_lambda * attention_l2_loss

        # Accuracy
        with tf.name_scope("attention_accuracy"):
            correct_predictions = tf.equal(self.attention_predictions, tf.argmax(self.input_y, 1))
            self.attention_correct_predictions = correct_predictions
            self.attention_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
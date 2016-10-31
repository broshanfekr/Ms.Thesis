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

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)



        with tf.name_scope("attention"):
            W_word_attention = tf.Variable(tf.random_normal([embedding_size, n_hidden_attention]), name="W_word_attention")
            b_word_attention = tf.Variable(tf.random_normal([n_hidden_attention]), name="b_word_attention"),
            attention_vector = tf.Variable(tf.random_normal([n_hidden_attention, 1]), name="attention_vector")

            l2_loss += tf.nn.l2_loss(W_word_attention)
            l2_loss += tf.nn.l2_loss(b_word_attention)
            l2_loss += tf.nn.l2_loss(attention_vector)


            tmp_input = tf.reshape(self.input_x, [-1, embedding_size])
            Ut = tf.tanh(tf.add(tf.matmul(tmp_input, W_word_attention), b_word_attention))

            doc_alfa = tf.exp(tf.matmul(Ut, attention_vector))

            doc_alfa = tf.reshape(doc_alfa, [-1, sequence_length])
            #doc_alfa = tf.mul(doc_alfa, doc_mask_list)

            Normalize_factor = tf.reduce_sum(doc_alfa, 1, keep_dims=True)
            Normalize_factor = tf.add(Normalize_factor, 1.0e-15)
            doc_alfa = tf.div(doc_alfa, Normalize_factor)
            tmp2_weight = tf.reduce_sum(doc_alfa, 1)

            doc_alfa = tf.reshape(doc_alfa, [-1, sequence_length, 1])
            self.weighted_input = tf.mul(self.input_x, doc_alfa)

            #self.h_pool_flat = tf.reduce_max(outputs, reduction_indices=1)



        # Embedding layer
        with tf.name_scope("embedding"):
            #W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            #self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_words_expanded = tf.expand_dims(self.weighted_input, -1)


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_words_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

            '''
            p = tf.Variable(tf.constant(0.5, shape=[num_filters_total, num_classes]), name = "p")
            final_val = tf.mul(x=W, y=p, name="final_val")

            normal_val = tf.nn.l2_normalize(x=W, dim=0)
            self.normalizing = tf.Variable.assign(self=W, value=normal_val*l2_constraint)
            self.last_layer_weight = W
            self.assine_final_value = tf.Variable.assign(self=W, value=final_val)
            '''


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.weigth_norm = l2_reg_lambda*l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.correct_predictions = correct_predictions
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

import numpy as np
import data_helpers
import os
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
###################################################################################################
x, y, seq_length = data_helpers.load_data()
# Split train/test set
x_train, x_dev = x[:-25000], x[-25000:]
y_train, y_dev = y[:-25000], y[-25000:]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
###################################################################################################

destfile = open("LSTM_Sentiment_result.txt", "w")
# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 100
Num_of_epochs = 10

# Network Parameters
n_input = 150 # embedding dimension
n_steps = seq_length # number of words in a sentence
n_hidden = 200 # hidden layer num of features
n_classes = 2 # total number of classes

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.all_variables())

def RNN(_X, _istate, _weights, _biases):

    _X = tf.transpose(_X, [1, 0, 2])
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    _X = tf.split(0, n_steps, _X)

    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    sum_outputs = tf.add_n(outputs)
    normal_output = tf.nn.l2_normalize(sum_outputs, dim = 0)

    # Linear activation
    return tf.matmul(normal_output, _weights['out']) + _biases['out']


def dev_step(x_batch, y_batch, writer=None):
    dev_baches = data_helpers.batch_iter(list(zip(x_batch, y_batch)), batch_size=batch_size, seq_length=seq_length,
                                         emmbedding_size=n_input, shuffle=False)
    total_loss = 0
    total_acc = 0
    index = 0
    total_correct_predictions = 0
    total_dev_data = 0
    for batch in dev_baches:
        if (batch.size == 0):
            continue
        batch_xs, batch_ys = zip(*batch)
        total_dev_data += len(batch_xs)

        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                            istate: np.zeros((batch_size, 2 * n_hidden))})
        # Calculate batch loss


        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                         istate: np.zeros((batch_size, 2 * n_hidden))})
        print("index is: ", index)
        print(", Minibatch Loss= " + "{:.6f}".format(loss) + ", testing Accuracy= " + "{:.5f}".format(acc))

        correct_predict = sess.run(prediction, feed_dict={x: batch_xs, y: batch_ys,
                                                            istate: np.zeros((batch_size, 2 * n_hidden))})
        s=""
        for i in correct_predict:
            s = s + str(i) + " "
        print(s)
        destfile.write(s)
        correct_predict = sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys,
                                                            istate: np.zeros((batch_size, 2 * n_hidden))})
        total_loss += loss
        total_acc += accuracy
        index += 1
        total_correct_predictions += np.sum(correct_predict)

    destfile.write("################################################################################################")
    avg_loss = total_loss / index
    avg_acc = total_acc / index
    real_acc = (total_correct_predictions * 1.0) / (total_dev_data)
    print("avarage_Loss: ", avg_loss, "\navarage_acc: ", avg_acc, "\nreal_acc: ", real_acc, "\n\n")


print("before pred")
pred = RNN(x, istate, weights, biases)
print("after pred")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
print("after cost")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
print("after optimizer")

# Evaluate model
prediction = tf.argmax(pred,1)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
print("after correct_pred")
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("after accuracy")

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the graph
with tf.Session() as sess:
    print("after with")
    sess.run(init)
    step = 1
    #dev_step(x_dev, y_dev)
    for epoch in range(Num_of_epochs):
        #path = saver.save(sess, checkpoint_prefix)
        #print("Saved model checkpoint to {}\n".format(path))

        print("epoch number: ", epoch)
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size=batch_size,seq_length=seq_length,
                                          emmbedding_size=n_input)
        step_number = 0
        for batch in batches:
            if(batch.size == 0):
                continue
            batch_xs, batch_ys = zip(*batch)
        # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2*n_hidden))})
            print("epoch:  " + str(epoch) + "      step:  " + str(step_number))
            #print(", Minibatch Loss= " + "{:.6f}".format(the_cost) + ", Training Accuracy= " + "{:.5f}".format(the_acc))
            step_number += 1
        #if ((epoch+1) % display_step == 0 or True):
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden))})

            correct_predict = sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys,
                                                                istate: np.zeros((batch_size, 2 * n_hidden))})
            sum_of_1label = np.sum(correct_predict)

            print(", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            print("sum of 1 label: " , sum_of_1label)

        if((epoch+1) % 2 == 0):
            dev_step(x_dev, y_dev)

    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("\nEvaluation:")
    dev_step(x_dev, y_dev)
    print("")

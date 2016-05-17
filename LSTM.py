import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import data_helpers
import os

import logging, sys, pprint

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
###################################################################################################
x, y, seq_length = data_helpers.load_data()
# Split train/test set
x_train, x_dev = x[:-25000], x[-25000:]
y_train, y_dev = y[:-25000], y[-25000:]

print("bero sucssecc")
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
###################################################################################################


# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 250
display_step = 100
Num_of_epochs = 10

# Network Parameters
n_input = 150 # MNIST data input (img shape: 28*28)
n_steps = seq_length # timesteps
n_hidden = 300 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)

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

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.all_variables())

def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    #suse=lstm_cell.state_size
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    dev_baches = data_helpers.dev_batch_iter(
        list(zip(x_batch, y_batch)), batch_size=batch_size, seq_length=seq_length,
        emmbedding_size=n_input)

    total_loss = 0
    total_acc = 0
    index = 0
    total_correct_predictions = 0
    total_dev_data = 0
    for batch in dev_baches:
        index += 1
        if (batch.size == 0):
            continue
        batch_xs, batch_ys = zip(*batch)
        total_dev_data += len(batch_xs)

        acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                            istate: np.zeros((batch_size, 2 * n_hidden))})
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                         istate: np.zeros((batch_size, 2 * n_hidden))})
        print(", Minibatch Loss= " + "{:.6f}".format(loss) + \
              ", testing Accuracy= " + "{:.5f}".format(acc))

        correct_predict = sess.run(correct_pred, feed_dict={x: batch_xs, y: batch_ys,
                                                            istate: np.zeros((batch_size, 2 * n_hidden))})

        total_loss += loss
        total_acc += accuracy
        total_correct_predictions += np.sum(correct_predict)
    avg_loss = total_loss / (index - 1)
    avg_acc = total_acc / (index - 1)
    real_acc = (total_correct_predictions * 1.0) / (total_dev_data)
    print("avg_loss {:g}, avg_acc {:g}, real_acc {:g}".format(avg_loss, avg_acc, real_acc))


print("before pred")
pred = RNN(x, istate, weights, biases)
print("after pred")

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
print("after cost")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
print("after optimizer")

# Evaluate model
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
    # Keep training until reach max iterations
    for epoch in range(Num_of_epochs):
        path = saver.save(sess, checkpoint_prefix)
        print("Saved model checkpoint to {}\n".format(path))

        print("epoch number: ", epoch)
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), batch_size=batch_size,
            seq_length=seq_length, emmbedding_size=n_input)
        step_number = 0
        for batch in batches:
            if(batch.size == 0):
                continue
            batch_xs, batch_ys = zip(*batch)
        # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       istate: np.zeros((batch_size, 2*n_hidden))})
            print("epoch:  " + str(epoch) + "      step:  " + str(step_number))
            step_number += 1
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2*n_hidden))})
            print(", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("\nEvaluation:")
    dev_step(x_dev, y_dev)
    print("")


    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
     #                                                        istate: np.zeros((test_len, 2*n_hidden))}))

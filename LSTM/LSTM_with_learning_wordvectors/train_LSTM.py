import numpy as np
import data_helpers
import os
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import LSTM_class

import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)



tf.flags.DEFINE_float("learning_rate", 0.001, "Dropout keep probability (default: 0.001)")
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 100)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("embedding_dim", 150, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("n_hidden", 200, "Dimensionality of LSTM hidden layer (default: 200")
tf.flags.DEFINE_integer("n_classes", 2, "Number of classes (default: 2")
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

destfile = open("LSTM_Sentiment_result.txt", "w")

FLAGS._parse_flags()
print("\nParameters:")
destfile.write("Parameters")
destfile.write("\n")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
    destfile.write("{}={}".format(attr.upper(), value))
    destfile.write("\n")
print("")

###################################################################################################
x, y, seq_length , vocabulary, vocabulary_inv = data_helpers.load_data()
# Split train/test set
x_train, x_dev = x[:-25000], x[-25000:]
y_train, y_dev = y[:-25000], y[-25000:]

myvocab_size = len(vocabulary)
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
###################################################################################################

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        myLSTM = LSTM_class.LSTM_class(
            sequence_length= seq_length,
            embedding_size= FLAGS.embedding_dim,
            n_classes= FLAGS.n_classes,
            n_hidden= FLAGS.n_hidden,
            vocab_size= myvocab_size)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(myLSTM.cost)  # Adam Optimizer

        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "mycheckpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "mymodel")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            myfeed_dict = {
              myLSTM.input_x: x_batch,
              myLSTM.input_y: y_batch,
              myLSTM.istate: np.zeros((FLAGS.batch_size, 2 * FLAGS.n_hidden))
            }
            _, loss, accuracy, currect_predict = sess.run([optimizer, myLSTM.cost, myLSTM.accuracy
                                                              ,myLSTM.correct_predictions], feed_dict= myfeed_dict)
            print("loss {:g}, acc {:g}".format(loss, accuracy))
            destfile.write("loss {:g}, acc {:g}".format(loss, accuracy))
            destfile.write("\n")


        def dev_step(x_batch, y_batch, writer=None):
            dev_baches = data_helpers.batch_iter(list(zip(x_batch, y_batch)), batch_size=FLAGS.batch_size,
                                                 seq_length=seq_length,
                                                 emmbedding_size=FLAGS.embedding_dim, shuffle=False)
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

                myfeed_dict = {
                    myLSTM.input_x: batch_xs,
                    myLSTM.input_y: batch_ys,
                    myLSTM.istate: np.zeros((FLAGS.batch_size, 2 * FLAGS.n_hidden))
                }

                acc, loss, correct_predict = sess.run([myLSTM.accuracy, myLSTM.cost, myLSTM.correct_predictions],
                                                      feed_dict= myfeed_dict)

                print("index is: ", index)
                print(" Minibatch Loss= " + "{:.6f}".format(loss) + ", testing Accuracy= " + "{:.5f}".format(acc))
                total_loss += loss
                total_acc += acc
                index += 1
                total_correct_predictions += np.sum(correct_predict)

            destfile.write(
                "################################################################################################")
            avg_loss = total_loss / index
            avg_acc = total_acc / index
            real_acc = (total_correct_predictions * 1.0) / (total_dev_data)
            print("avarage_Loss: ", avg_loss, "    avarage_acc: ", avg_acc, "   real_acc: ", real_acc, "\n\n")


        for epoch in range(FLAGS.num_epochs):
            print("epoch number: ", epoch)
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size=FLAGS.batch_size,
                                              seq_length=seq_length, emmbedding_size=FLAGS.embedding_dim)

            step_number = 0
            for batch in batches:
                if (batch.size == 0):
                    continue
                batch_xs, batch_ys = zip(*batch)
                # Fit training using batch data
                train_step(batch_xs, batch_ys)
                print("epoch:  " + str(epoch) + "      step:  " + str(step_number))
                step_number += 1

            if ((epoch + 1) % FLAGS.checkpoint_every == 0):
                path = saver.save(sess, checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))

        path = saver.save(sess, checkpoint_prefix)
        print("Saved model checkpoint to {}\n".format(path))
        print("Optimization Finished!")
        # Calculate accuracy for 256 mnist test images
        print("\nEvaluation:")
        dev_step(x_dev, y_dev)
        print("")
destfile.close()
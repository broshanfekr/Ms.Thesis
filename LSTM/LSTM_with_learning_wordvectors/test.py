import numpy as np
import os
import time
import datetime
import data_helpers
import tensorflow as tf


# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/mycheckpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("n_hidden", 200, "Dimensionality of LSTM hidden layer (default: 200")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


destfile = open("CNN_Sentiment_result.txt", "w")

# Load data. Load your own data here
print("Loading data...")
destfile.write("Loading data...")
destfile.write("\n")
x, y, seq_length , vocabulary, vocabulary_inv = data_helpers.load_data()
# Split train/test set
x_train, x_dev = x[:-25000], x[-25000:]
y_train, y_dev = y[:-25000], y[-25000:]


print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
destfile.write("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
destfile.write("\n")

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        istate = graph.get_operation_by_name("istate").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        #dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), batch_size=100,
                                                 seq_length=seq_length,emmbedding_size=150, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        iternum = 0
        for x_test_batch in batches:
            if (x_test_batch.size == 0):
                continue
            print('iternum is :  ', iternum)
            iternum+=1
            x_batch, y_batch = zip(*x_test_batch)
            myfeed_dict = {
              input_x: x_batch,
              input_y: y_batch,
              istate: np.zeros((FLAGS.batch_size, 2 * FLAGS.n_hidden))
            }
            batch_predictions = sess.run(predictions, myfeed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy
y_test = np.argmax(y_dev, axis=1)
correct_predictions = float(sum(all_predictions == y_test))
print("Total number of test examples: {}".format(len(y_test)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
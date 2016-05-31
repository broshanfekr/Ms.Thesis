import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import tensorflow as tf

# Parameters
# ==================================================
# Model Hyperparameters
#"5,6,7,9"
tf.flags.DEFINE_integer("embedding_dim", 150, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 150, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("l2_constraint", 3.0, "l2 constraint s")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


destfile = open("CNN_Sentiment_result.txt", "w")

FLAGS._parse_flags()
print("\nParameters:")
destfile.write("Parameters")
destfile.write("\n")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
    destfile.write("{}={}".format(attr.upper(), value))
    destfile.write("\n")
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
destfile.write("Loading data...")
destfile.write("\n")
x, y, seq_length = data_helpers.load_data()

#x, y, vocabulary, vocabulary_inv = data_helpers_orginal.load_data()
# Randomly shuffle data
#np.random.seed(10)
#shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x#[shuffle_indices]
y_shuffled = y#[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-25000], x_shuffled[-25000:]
y_train, y_dev = y_shuffled[:-25000], y_shuffled[-25000:]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
destfile.write("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
destfile.write("\n")

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=seq_length,
            num_classes=2,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            l2_constraint = FLAGS.l2_constraint)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        #myx1 = cnn.last_layer_weight.eval(sess)
        #sess.run(cnn.normalizing)
        #myx2 = cnn.last_layer_weight.eval(sess)
        #print(np.linalg.norm(myx1[:, 0]))
        #print(np.linalg.norm(myx2[:, 0]))


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, correct_predict = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.correct_predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: train-step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            destfile.write("{}: train-step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            destfile.write("\n")
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            dev_baches = data_helpers.batch_iter(list(zip(x_batch, y_batch)), batch_size=FLAGS.batch_size,
                                                 seq_length=seq_length,emmbedding_size=FLAGS.embedding_dim, shuffle=False)

            total_loss = 0
            total_acc = 0
            index = 0
            total_correct_predictions = 0
            total_dev_data = 0
            for batch in dev_baches:
                index += 1
                if (batch.size == 0):
                    continue
                x_batch, y_batch = zip(*batch)
                total_dev_data += len(x_batch)
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, correct_predict = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.correct_predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: dev-step {}, loss {:g}, acc {:g}".format(time_str, index, loss, accuracy))
                destfile.write(str(time_str) + "  dev-step: " + str(index) + "  acc: " + str(accuracy))
                destfile.write("\n")
                total_loss += loss
                total_acc += accuracy
                total_correct_predictions += np.sum(correct_predict)
            avg_loss = total_loss / (index - 1)
            avg_acc = total_acc / (index - 1)
            real_acc = (total_correct_predictions*1.0) / (total_dev_data)
            print("avg_loss {:g}, avg_acc {:g}, real_acc {:g}".format(avg_loss, avg_acc, real_acc))
            destfile.write("avg_loss {:g}, avg_acc {:g}, real_acc {:g}".format(avg_loss, avg_acc, real_acc))
            destfile.write("\n")
            if writer:
                writer.add_summary(summaries, step)

        for epoch in range(FLAGS.num_epochs):
            print("epoch number is: ", epoch)
            destfile.write("epoch number is: "+ str(epoch))
            destfile.write("\n")
            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size=FLAGS.batch_size,
                                              seq_length=seq_length, emmbedding_size=FLAGS.embedding_dim)
            # Training loop. For each batch...
            for batch in batches:
                if(batch.size == 0):
                    continue
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)

                #sess.run(cnn.normalizing)
                myx2 = cnn.last_layer_weight.eval(sess)
                print("normalaized_weigth: ", np.linalg.norm(myx2[:, 0]))


                current_step = tf.train.global_step(sess, global_step)

            if ((epoch+1) % FLAGS.checkpoint_every == 0):
                path = saver.save(sess, checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))

            if ((epoch+1) % FLAGS.evaluate_every) == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")

        print("assining final value")
        sess.run(cnn.assine_final_value)
        myx2 = cnn.last_layer_weight.eval(sess)
        print("normalaized_weigth: ", np.linalg.norm(myx2[:, 0]))

        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
        destfile.close()

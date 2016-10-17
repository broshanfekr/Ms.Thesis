import numpy as np
import data_helpers
import os
import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
import LSTM_class
from sklearn.cross_validation import train_test_split

import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


############################### creating a parser for arguments###################################
import argparse
def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--input_dataset_path', '-input_dataset_path', type=str, default="input_data/total.tar",
        metavar='PATH',
        help="Path of the dataset"
    )
    parser.add_argument(
        '--word2vec_model_path', '-word2vec_model_path', type=str, default="word2vec_model/IMDB_skipgram_model",
        metavar='PATH',
        help="the path of trained word2vec model."
    )
    parser.add_argument(
        '--is_separated', '-separate', type=bool, default=False,
        help="does the dataset have official train/test split or not?"
    )
    parser.add_argument(
        '--output_path', '-output_path', type=str, default="LSTM_Sentiment_result.txt",
        metavar='PATH',
        help="output file that contain logs during train and test"
    )

    ############################# model hyper parameters #################################
    parser.add_argument(
        '--max_seq_len_cutoff', '-max_seq_len_cutoff', type=int, default=1350,
        help="the max len of a sentence."
    )
    parser.add_argument(
        '--learning_rate', '-learning_rate', type=float, default=0.001,
        help="learning rate parameter"
    )
    parser.add_argument(
        '--n_classes', '-n_classes', type=int, default=2,
        help="number of classes in classification problem"
    )
    parser.add_argument(
        '--batch_size', '-batch_size', type=int, default=100,
        help="size of the batch in training step"
    )
    parser.add_argument(
        '--num_epochs', '-num_epochs', type=int, default=30,
        help="defines the number of training epochs."
    )
    parser.add_argument(
        '--embedding_dim', '-embedding_dim', type=int, default=150,
        help="Dimensionality of word embedding"
    )
    parser.add_argument(
        '--n_hidden', '-n_hidden', type=int, default=200,
        help="Dimensionality of LSTM hidden layer"
    )
    parser.add_argument(
        '--n_hidden_attention', '-n_hidden_attention', type=int, default=100,
        help="Dimensionality of attention hidden layer"
    )
    parser.add_argument(
        '--evaluate_every', '-evaluate_every', type=int, default=2,
        help="Evaluate model on dev set after this many steps"
    )
    parser.add_argument(
        '--checkpoint_every', '-checkpoint_every', type=int, default=1,
        help="Save model after this many steps"
    )
    parser.add_argument(
        '--l2_reg_lambda', '-l2_reg_lambda', type=float, default=0.00003,
        help="L2 regularizaion lambda"
    )
    parser.add_argument(
        '--dropout_keep_prob', '-dropout_keep_prob', type=float, default=0.5,
        help="Dropout keep probability (default: 0.5)"
    )
    parser.add_argument(
        '--checkpoint_dir', '-checkpoint_dir', type=str, default="runs/checkpoints",
        help="Checkpoint directory from training run"
    )
    parser.add_argument(
        '--allow_soft_placement', '-allow_soft_placement', type=bool, default=True,
        help="Allow device soft device placement"
    )
    parser.add_argument(
        '--log_device_placement', "-log_device_placement", type=bool, default=False,
        help="Log placement of ops on devices"
    )
    parser.add_argument(
        '--is_load_model', '-is_load_model', type=bool, default=False,
        help="do we want to load previes model?"
    )

    return parser
##################################################################################################
parser = create_parser()
FLAGS = parser.parse_args()
destfile = open(FLAGS.output_path, "w")


print("\nParameters:")
destfile.write("Parameters")
destfile.write("\n")
for arg in vars(FLAGS):
    print("{} : {}".format(arg, getattr(FLAGS, arg)))
    destfile.write("{} : {}".format(arg, getattr(FLAGS, arg)))
    destfile.write("\n\n")
print("")
##############################   Reading Data   ###################################################
def read_from_dataset(dataset_path, word2vec_model_path, n_classes, max_seq_len_cutoff):
    print("Loading data...")
    x, y, doc_length, sent_length, word2vec_vocab, word2vec_vec = data_helpers.load_data(
        dataset_path, word2vec_model_path, n_classes, max_seq_len_cutoff)

    if (dataset_path.split("/")[-1] == "total.tar"):
        x_train, x_dev = x[:-25000], x[-25000:]
        y_train, y_dev = y[:-25000], y[-25000:]
    else:
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.3, random_state=42)

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    seq_max_len = [doc_length, sent_length]
    return x_train, x_dev, y_train, y_dev, seq_max_len, word2vec_vocab, word2vec_vec
##################################################################################################
def My_main():
    x_train, x_dev, y_train, y_dev, seq_max_len, word2vec_vocab, word2vec_vec = read_from_dataset(
        FLAGS.input_dataset_path, FLAGS.word2vec_model_path, FLAGS.n_classes, FLAGS.max_seq_len_cutoff)

    ##############################  Variable Definition ##############################################
    input_x = tf.placeholder(tf.float32, [None, seq_max_len[0], seq_max_len[1], FLAGS.embedding_dim], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name='input_y')
    doc_len_list = tf.placeholder(tf.int64, [None], name='seq_len_list')
    sent_len_list = tf.placeholder(tf.int64, [None], name='seq_len_list')
    sent_mask_list = tf.placeholder(tf.float32, [None, seq_max_len[1]], name='sent_mask_list')
    doc_mask_list = tf.placeholder(tf.float32, [None, seq_max_len[0]], name='doc_mask_list')
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    weights = {
        'W_hidden': tf.Variable(tf.random_normal([FLAGS.embedding_dim, FLAGS.n_hidden]), name='W_hidden'),
        'W_word_attention' : tf.Variable(tf.random_normal([2*FLAGS.n_hidden, FLAGS.n_hidden_attention]),
                                         name="W_word_attention"),
        'word_attention_vector' : tf.Variable(tf.random_normal([FLAGS.n_hidden_attention, 1]),
                                              name="word_attention_weigth"),
        'W_sent_attention': tf.Variable(tf.random_normal([2*FLAGS.n_hidden, FLAGS.n_hidden_attention]),
                                        name="W_sent_attention"),
        'sent_attention_vector': tf.Variable(tf.random_normal([FLAGS.n_hidden_attention, 1]),
                                             name="sent_attention_weigth"),
        'W_out' : tf.Variable(tf.random_normal([2*FLAGS.n_hidden, FLAGS.n_classes]), name='W_out')
    }
    biases = {
        'B_hidden': tf.Variable(tf.random_normal([FLAGS.n_hidden]), name='B_hidden'),
        'b_word_attention' : tf.Variable(tf.random_normal([FLAGS.n_hidden_attention]), name="b_word_attention"),
        'b_sent_attention' : tf.Variable(tf.random_normal([FLAGS.n_hidden_attention]), name="b_sent_attention"),
        'B_out' : tf.Variable(tf.random_normal([FLAGS.n_classes]), name='B_out')
    }

    with tf.name_scope('Model'):
        myLSTM_outputs = LSTM_class.dynamic_bidirectional_LSTM(input_x,
                                                               FLAGS.embedding_dim,
                                                               seq_max_len,
                                                               doc_len_list,
                                                               sent_len_list,
                                                               FLAGS.n_hidden,
                                                               weights,
                                                               biases,
                                                               sent_mask_list,
                                                               doc_mask_list,
                                                               dropout_keep_prob)

    with tf.name_scope('Predict'):
        predictions = tf.argmax(myLSTM_outputs, 1, name='predictions')

    with tf.name_scope("loss"):
        weight_amount = tf.nn.l2_loss(weights['W_hidden']) + tf.nn.l2_loss(weights['W_out']) \
                        + tf.nn.l2_loss(biases['B_hidden']) + tf.nn.l2_loss(biases['B_out']) \
                        + tf.nn.l2_loss(weights['W_word_attention']) + tf.nn.l2_loss(weights['word_attention_vector']) \
                        + tf.nn.l2_loss(weights['W_sent_attention']) + tf.nn.l2_loss(weights['sent_attention_vector'])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(myLSTM_outputs, input_y)) \
               + FLAGS.l2_reg_lambda * weight_amount

    with tf.name_scope('Accuracy'):
        correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    #with tf.name_scope('SGD'):
    #    # Gradient Descent
    #    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    #    # Op to calculate every variable gradient
    #    grads = tf.gradients(loss, tf.trainable_variables())
    #    grads = list(zip(grads, tf.trainable_variables()))
    #    # Op to update all variables according to their gradient
    #    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

    with tf.name_scope('Optimizer'):
        optimize = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)# Adam Optimizer

    ##################################################################################################
    ################################# make summary ###################################################
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs2"))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "mymodel")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    #saver = tf.train.Saver(tf.all_variables())
    saver = tf.train.Saver()

    # Summaries for loss and accuracy
    loss_summary = tf.scalar_summary("loss", loss)
    acc_summary = tf.scalar_summary("accuracy", accuracy)

    # Train Summaries
    train_summary_op = tf.merge_summary([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")

    # Dev summaries
    dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")

    all_summary_dir = os.path.join(out_dir, "summaries", "all_summaries")

    for var in tf.trainable_variables():
        tf.histogram_summary(var.name, var)
    # Summarize all gradients
    #for grad, var in grads:
    #    tf.histogram_summary(var.name + '/gradient', grad)
    # Merge all summaries into a single op
    merged_summary_op = tf.merge_all_summaries()

    ############################# end summary ########################################################
    ##################################################################################################
    if(FLAGS.is_load_model == True):
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        my_train_step = 0
        my_dev_step = 0

        sess.run(init)

        if(FLAGS.is_load_model == True):
            load_path = saver.restore(sess, checkpoint_file)
            pass
        # op to write logs to Tensorboard

        all_summary_writer = tf.train.SummaryWriter(all_summary_dir, graph=tf.get_default_graph())
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, graph=tf.get_default_graph())
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, graph=tf.get_default_graph())

        def build_mask(batch_seq_len):
            sent_mask = []
            for seqlen in batch_seq_len[1]:
                tmp_list=[]
                for i in range(seq_max_len[1]):
                    if i < seqlen:
                        tmp_list.append(1)
                    else:
                        tmp_list.append(0)
                sent_mask.append(tmp_list)
            sent_mask = np.asarray(sent_mask)

            doc_mask = []
            for seqlen in batch_seq_len[0]:
                tmp_list = []
                for i in range(seq_max_len[0]):
                    if i < seqlen:
                        tmp_list.append(1)
                    else:
                        tmp_list.append(0)
                doc_mask.append(tmp_list)
            doc_mask = np.asarray(doc_mask)
            return sent_mask, doc_mask

        def dev_step(x_dev, y_dev, test_set = "dev", writer=None):
            dev_baches = data_helpers.batch_iter(list(zip(x_dev, y_dev)),
                                                 batch_size=FLAGS.batch_size,
                                                 seq_length=seq_max_len,
                                                 emmbedding_size=FLAGS.embedding_dim,
                                                 word2vec_vocab=word2vec_vocab,
                                                 word2vec_vec=word2vec_vec,
                                                 is_shuffle=False)
            total_loss = 0
            total_acc = 0
            index = 0
            total_correct_predictions = 0
            total_dev_data = 0
            for batch in dev_baches:
                if (len(batch[0]) == 0):
                    continue
                x_batch, y_batch = zip(*batch[0])
                batch_seq_len = batch[1]
                total_dev_data += len(x_batch)

                sent_mask, doc_mask = build_mask(batch_seq_len)

                myfeed_dict = {
                    input_x: x_batch,
                    input_y: y_batch,
                    doc_len_list: batch_seq_len[0],
                    sent_len_list: batch_seq_len[1],
                    sent_mask_list : sent_mask,
                    doc_mask_list : doc_mask,
                    dropout_keep_prob: 1
                }

                acc, cost, correct_predict, summaries = sess.run(
                    [accuracy,
                     loss,
                     correct_predictions,
                     dev_summary_op],
                    feed_dict= myfeed_dict)

                if(test_set == "dev"):
                    dev_summary_writer.add_summary(summaries, index + my_dev_step * 250)
                print("on {}, test index: {:g}, Minibatch Loss: {:.6f}, acc: {:g}".format(test_set, index, cost, acc))
                destfile.write("on {}, test index: {:g}, Minibatch Loss: {:.6f}, acc: {:g}\n".format(test_set, index, cost, acc))
                total_loss += cost
                total_acc += acc
                index += 1
                total_correct_predictions += np.sum(correct_predict)
            print("######################################################################")
            destfile.write("####################################################################\n")
            avg_loss = total_loss / index
            avg_acc = total_acc / index
            real_acc = (total_correct_predictions * 1.0) / (total_dev_data)
            print("on {}, avarage_Loss: {:g}, avarage_acc: {:.6f}, real_acc: {:g}\n".format(test_set, avg_loss, avg_acc, real_acc))
            destfile.write("on {}, avarage_Loss: {:g}, avarage_acc: {:.6f}, real_acc: {:g}\n\n".format(test_set, avg_loss, avg_acc, real_acc))

        def train_step(x_batch, y_batch, batch_seq_len, my_train_step, epoch_num):

            sent_mask, doc_mask = build_mask(batch_seq_len)

            myfeed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                doc_len_list : batch_seq_len[0],
                sent_len_list : batch_seq_len[1],
                sent_mask_list : sent_mask,
                doc_mask_list : doc_mask,

                dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            '''
            suse_vars = sess.run([my_vars], feed_dict=myfeed_dict)
            suse_vars = suse_vars[0]
            embedded_words,\
            hidden_matrix,\
            sent_output,\
            first_alfa,\
            sent_alfa,\
            tmp_weight,\
            weigthed_sent_outputs,\
            sent_lstm_outputs,\
            doc_output,\
            doc_alfa,\
            tmp2_weight,\
            final_output,\
            output_layer = suse_vars
            '''

            _, cost, acc, currect_predict, weigth_norm, summaries, all_summaries = sess.run(
                [optimize,
                 loss,
                 accuracy,
                 correct_predictions,
                 weight_amount,
                 train_summary_op,
                 merged_summary_op],
                feed_dict=myfeed_dict)

            train_summary_writer.add_summary(summaries, my_train_step)
            all_summary_writer.add_summary(all_summaries, my_train_step)
            weigth_norm = weigth_norm * FLAGS.l2_reg_lambda
            print("epoch: {:g}, iteration: {:g}, weigth_norm: {:.6f},  loss: {:.4f}, acc: {:g}".format(epoch_num, my_train_step, weigth_norm, cost, acc))
            destfile.write("epoch: {:g}, iteration: {:g}, weigth_norm: {:.6f},  loss: {:.4f}, acc: {:g}\n".format(epoch_num, my_train_step, weigth_norm, cost, acc))

        #dev_step(x_train, y_train, test_set="train")
        for epoch in range(FLAGS.num_epochs):
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)),
                                              batch_size=FLAGS.batch_size,
                                              seq_length=seq_max_len,
                                              emmbedding_size=FLAGS.embedding_dim,
                                              word2vec_vocab=word2vec_vocab,
                                              word2vec_vec=word2vec_vec)

            for batch in batches:
                my_train_step += 1
                if (len(batch[0]) == 0):
                    continue
                batch_xs, batch_ys = zip(*batch[0])
                batch_seq_len = batch[1]
                train_step(batch_xs, batch_ys, batch_seq_len, my_train_step, epoch)

            if((epoch + 1) % FLAGS.evaluate_every == 0):
                print("testing on dev set: ")
                destfile.write("testing on dev set:\n")
                dev_step(x_dev, y_dev)
                my_dev_step += 1
                print("###############################################")
                destfile.write("###############################################\n")
                print("testing on train set: ")
                destfile.write("testing on train set: \n")
                dev_step(x_train, y_train, test_set="train")

            if ((epoch + 1) % FLAGS.checkpoint_every == 0):
                path = saver.save(sess, checkpoint_prefix, global_step=(epoch+1))
                print("Saved model checkpoint to {}\n".format(path))


        path = saver.save(sess, checkpoint_prefix,global_step=20)
        print("Saved model checkpoint to {}\n".format(path))
        print("Optimization Finished!")
        print("\nEvaluation:")
        dev_step(x_dev, y_dev)
        print("")

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    My_main()
    stop = timeit.default_timer()

    spent_time = int(stop - start)
    sec = spent_time % 60
    spent_time = int(spent_time / 60)
    minute = spent_time % 60
    spent_time = int(spent_time / 60)
    hours = spent_time
    print("h: ", hours, "  minutes: ", minute, "  secunds: ", sec)
    destfile.write("\nh: " + str(hours) + "  minutes: " + str(minute) + "  secunds: " + str(sec) + "\n")
    destfile.close()


#tensorboard --logdir

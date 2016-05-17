import numpy as np
import re
import itertools
from collections import Counter
import tarfile
from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
import gensim
import copy


def Load_Model(name='./myIMDB_model.d2v'):
    return Doc2Vec.load(name)


def clean_str(review_docs, method=2):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    output_docs = []
    if(method == 1):
        for string in review_docs:
            string = BeautifulSoup(string, "lxml").get_text()
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            string = string.strip().lower()
            string = string.split(" ")
            output_docs.append(string)
    elif(method==2):
        for string in review_docs:
            words = gensim.utils.to_unicode(string).split()
            output_docs.append(words)
    return output_docs


def Load_IMDB_Data_and_Label(is_remove_stopwords = False):
    dataset_file_name = "total.tar"
    tar = tarfile.open(dataset_file_name)
    for member in tar.getmembers():
        file_name = member.name.split("-")
        isTrain = file_name[0]
        data_label = file_name[-1].split(".")[0]
        f = tar.extractfile(member)
        content = f.readlines()
        f.close()
        if(isTrain == "train"):
            if(data_label == "pos"):
                positive_examples = content
                print('posetive data parsed from training set')
            elif(data_label == "neg"):
                negative_examples = content
                print("negative data parsed from training set")
            else:
                pass
        elif(isTrain == "test"):
            if(data_label == "pos"):
                test_positive_examples = content
                print("test positive examples loaded")
            elif(data_label == "neg"):
                test_negative_examples = content
                print("negative sentences parsed from testing set")
    tar.close()

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    test_positive_examples = [s.strip() for s in test_positive_examples]
    test_negative_examples = [s.strip() for s in test_negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    test_x_text = test_positive_examples + test_negative_examples

    x_text = clean_str(x_text)#[clean_str(sent) for sent in x_text]
    #x_text = [s.split(" ") for s in x_text]

    test_x_text = clean_str(test_x_text)#[clean_str(sent) for sent in test_x_text]
    #test_x_text = [s.split(" ") for s in test_x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]

    test_positive_labels = [[0, 1] for _ in test_positive_examples]
    test_negative_labels = [[1, 0] for _ in test_negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)
    test_y = np.concatenate([test_positive_labels, test_negative_labels], 0)
    return [x_text, y, test_x_text, test_y]


def pad_sentences(sentences, test_sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    sequence_length1 = max(len(x) for x in test_sentences)
    sequence_length = max(sequence_length, sequence_length1)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)


    test_sentences_padded = []
    for i in range(len(test_sentences)):
        sentence = test_sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        test_sentences_padded.append(new_sentence)

    return padded_sentences, test_sentences_padded, sequence_length


def build_input_data(sentences, labels):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    doc2vec_Model = Load_Model(name='simple_model0')
    doc2vec_vocab = doc2vec_Model.vocab
    doc2vec_vec = doc2vec_Model.syn0

    X_data = []
    for line in sentences:
        flag = False
        doc_matrix = []
        for word in line:
            #word = vocabulary_inv[index]
            try:
                doc2vec_index = doc2vec_vocab[word].index
                doc2vec_vector = doc2vec_vec[doc2vec_index]
            except:
                if(word == "<PAD/>"):
                    if(flag == False):
                        doc2vec_vector = np.zeros(doc2vec_vec.shape[1])
                        flag = True
                    else:
                        break
                else:
                    doc2vec_vector = np.random.uniform(low=-0.5, high=0.5, size=doc2vec_vec.shape[1])
            doc_matrix.append(doc2vec_vector)
        X_data.append(doc_matrix)
    X_data = np.asarray(X_data)
    y = np.array(labels)
    return [X_data, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, test_sentences , test_labels = Load_IMDB_Data_and_Label()
    sentences_padded, test_sentences_padded, seq_length = pad_sentences(sentences, test_sentences)
    x, y = build_input_data(sentences_padded + test_sentences_padded, np.concatenate([labels, test_labels], 0))
    return [x, y, seq_length]


def batch_iter(data, batch_size, seq_length, emmbedding_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1


    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        tmp_data = copy.deepcopy(shuffled_data[start_index:end_index])
        for x in tmp_data:
            if(len(x[0]) < seq_length):
                num_padding = seq_length - len(x[0])
                x_bar = np.zeros([num_padding, emmbedding_size])
                x[0] = np.concatenate([x[0], x_bar], axis=0)
        yield tmp_data

def dev_batch_iter(data , batch_size, seq_length, emmbedding_size, shuffle = False):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    print("processing evaluatin data...")
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        tmp_data = copy.deepcopy(shuffled_data[start_index:end_index])
        for x in tmp_data:
            if (len(x[0]) < seq_length):
                num_padding = seq_length - len(x[0])
                x_bar = np.zeros([num_padding, emmbedding_size])
                x[0] = np.concatenate([x[0], x_bar], axis=0)
        yield tmp_data

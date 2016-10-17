import numpy as np
import re
import itertools
from collections import Counter
import tarfile
# from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
import gensim
import copy
from nltk.tokenize import wordpunct_tokenize
from gensim.models.word2vec import Word2Vec
from nltk import tokenize
import random


def Load_Model(name='./myIMDB_model.d2v'):
    return Doc2Vec.load(name)


def clean_str(review_docs, max_seq_len_cutoff, is_decode=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    output_docs = []
    for string in review_docs:
        if (is_decode == True):
            string = string.decode("utf-8")
        else:
            pass


        string = string.lower()


        susegar = wordpunct_tokenize(string)
        if(len(susegar) > max_seq_len_cutoff):
            continue
            words = words[:max_seq_len_cutoff]
            string = ' '.join(word for word in words)


        sentences = tokenize.sent_tokenize(string)
        sent_words = []
        for sent in sentences:
            if (len(sent) == 1):
                continue
            words = wordpunct_tokenize(sent)
            for index, w in enumerate(words):
                if (w.replace('.', '', 1).isdigit()):
                    words[index] = '<num>'
            sent_words.append(words)
        output_docs.append(sent_words)
    return output_docs


def Load_IMDB_Data_and_Label(dataset_file_name="total.tar", max_seq_len_cutoff=1350, is_remove_stopwords=False):
    tar = tarfile.open(dataset_file_name)
    for member in tar.getmembers():
        file_name = member.name.split("-")
        isTrain = file_name[0]
        data_label = file_name[-1].split(".")[0]
        f = tar.extractfile(member)
        content = f.readlines()
        f.close()
        if (isTrain == "train"):
            if (data_label == "pos"):
                positive_examples = content
                print('posetive data parsed from training set')
            elif (data_label == "neg"):
                negative_examples = content
                print("negative data parsed from training set")
            else:
                pass
        elif (isTrain == "test"):
            if (data_label == "pos"):
                test_positive_examples = content
                print("test positive examples loaded")
            elif (data_label == "neg"):
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

    x_text = clean_str(x_text, max_seq_len_cutoff)  # [clean_str(sent) for sent in x_text]
    # x_text = [s.split(" ") for s in x_text]

    test_x_text = clean_str(test_x_text, max_seq_len_cutoff)  # [clean_str(sent) for sent in test_x_text]
    # test_x_text = [s.split(" ") for s in test_x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]

    test_positive_labels = [[0, 1] for _ in test_positive_examples]
    test_negative_labels = [[1, 0] for _ in test_negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)
    test_y = np.concatenate([test_positive_labels, test_negative_labels], 0)
    return [x_text, y, test_x_text, test_y]

def max_seq_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    doc_length = max(len(x) for x in sentences)

    sent_length = 0
    for sent in sentences:
        for x in sent:
            if (len(x) > sent_length):
                sent_length = len(x)

    return doc_length, sent_length

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def build_input_data_from_word2vec(sentence, word2vec_vocab, word2vec_vec):
    """
    Maps sentenc and vectors based on a word2vec model.
    """
    X_data = []
    for word in sentence:
        try:
            word2vec_index = word2vec_vocab[word].index
            word_vector = word2vec_vec[word2vec_index]
        except:
            word_vector = np.random.uniform(low=-0.25, high=0.25, size=word2vec_vec.shape[1])
        X_data.append(word_vector)
    X_data = np.asarray(X_data)
    return X_data

def load_data(dataset_path, word2vec_model_path, n_class=2, max_seq_len_cutoff=1350):
    """
    Loads and preprocessed data from dataset file.
    """
    if (dataset_path.split("/")[-1] == "total.tar"):
        sentences, labels, test_sentences, test_labels = Load_IMDB_Data_and_Label(dataset_path, max_seq_len_cutoff)
        x_text = sentences + test_sentences
        y = np.concatenate([labels, test_labels], 0)
    else:
        dataset_file = open(dataset_path, "r")
        dataset_content = dataset_file.readlines()

        x_text = []
        y = []
        for element in dataset_content:
            element = element.lower()
            element = element.split("\t")
            label = int(element[0])
            text = element[1].strip()
            if (len(text) == 0):
                continue
            x_text.append(text)
            tmp_lable = np.zeros(n_class)
            if (n_class == 2):
                tmp_lable[label] = 1
            else:
                tmp_lable[label - 1] = 1
            y.append(tmp_lable)

        x_text = clean_str(x_text, max_seq_len_cutoff, is_decode=False)

    y = np.asarray(y)
    doc_length = max(len(x) for x in x_text)

    sent_length = 0
    for sent in x_text:
        for x in sent:
            if (len(x) > sent_length):
                sent_length = len(x)

    word2vec_Model = Load_Model(word2vec_model_path)
    word2vec_vocab = word2vec_Model.vocab
    word2vec_vec = word2vec_Model.syn0
    return [x_text, y, doc_length, sent_length, word2vec_vocab, word2vec_vec]


def batch_iter(data, batch_size, seq_length, emmbedding_size, word2vec_vocab, word2vec_vec, is_shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1

    # Shuffle the data at each epoch
    if is_shuffle:
        random.shuffle(data)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        tmp_data = copy.deepcopy(data[start_index:end_index])

        sent_seq_len = []
        doc_seq_len = []
        tmp_docs = []
        tmp_labels = []
        for doc in tmp_data:
            tmp_sent = []
            doc_seq_len.append(len(doc[0]))
            for sent in doc[0]:

                sent_seq_len.append(len(sent))
                sent_vector = build_input_data_from_word2vec(sent, word2vec_vocab, word2vec_vec)
                if (len(sent_vector) < seq_length[1]):
                    num_padding = seq_length[1] - len(sent)
                    x_bar = np.zeros([num_padding, emmbedding_size])
                    sent_vector = np.concatenate([sent_vector, x_bar], axis=0)
                tmp_sent.append(sent_vector)
            if (len(tmp_sent) < seq_length[0]):
                num_padding = seq_length[0] - len(tmp_sent)
                x_bar = np.zeros([num_padding, seq_length[1], emmbedding_size])
                tmp_sent = np.concatenate([tmp_sent, x_bar], axis=0)
                for i in range(num_padding):
                    sent_seq_len.append(0)
            tmp_docs.append(tmp_sent)
            tmp_labels.append(doc[1])
        tmp_docs = np.asarray(tmp_docs)
        tmp_labels = np.asarray(tmp_labels)

        tmp_data = list(zip(tmp_docs, tmp_labels))
        batch_seq_len = [doc_seq_len, sent_seq_len]
        yield [tmp_data, batch_seq_len]

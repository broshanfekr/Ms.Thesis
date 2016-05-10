import numpy as np
import re
import itertools
from collections import Counter
import tarfile
from bs4 import BeautifulSoup



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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
    return string.strip().lower()



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
                print("test negative examples loaded")
    tar.close()
    print("extracting positive examples...")
    positive_examples = [s.strip() for s in positive_examples]
    print("extracting negative examples...")
    negative_examples = [s.strip() for s in negative_examples]
    print("extracting positive test examples...")
    test_positive_examples = [s.strip() for s in test_positive_examples]
    print("extracting negative test examples...")
    test_negative_examples = [s.strip() for s in test_negative_examples]
    print("splitting by word...")
    # Split by words
    x_text = positive_examples + negative_examples
    test_x_text = test_positive_examples + test_negative_examples

    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    test_x_text = [clean_str(sent) for sent in test_x_text]
    test_x_text = [s.split(" ") for s in test_x_text]
    print("generating labels...")
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]

    test_positive_labels = [[0, 1] for _ in test_positive_examples]
    test_negative_labels = [[1, 0] for _ in test_negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)
    test_y = np.concatenate([test_positive_labels, test_negative_labels], 0)
    return [x_text, y, test_x_text, test_y]

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


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

    return padded_sentences, test_sentences_padded

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


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, test_sentences , test_labels = Load_IMDB_Data_and_Label()
    sentences_padded, test_sentences_padded = pad_sentences(sentences, test_sentences)
    #test_sentences_padded = pad_sentences(test_sentences)

    vocabulary, vocabulary_inv = build_vocab(sentences_padded + test_sentences_padded)
    x, y = build_input_data(sentences_padded + test_sentences_padded, np.concatenate([labels, test_labels], 0), vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        print("epoch number is: ", epoch)
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            if(batch_num == num_batches_per_epoch-1):
                mysusegar = 5
                pass
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def dev_batch_iter(data , batch_size, num_epochs=1, shuffle = False):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        print("epoch number is: ", epoch)
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

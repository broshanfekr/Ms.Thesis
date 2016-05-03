from __future__ import unicode_literals
from hazm import Normalizer
from hazm import sent_tokenize, word_tokenize
from hazm import Stemmer, Lemmatizer
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
import numpy as np
stemmer = Stemmer()
normalizer = Normalizer()
################## define variables-----------------------------------------------
num_features=100
num_Of_epoch = 0
train_rate = 0.6
validate_rate = 0.1
sentences = []  # Initialize an empty list of sentences
mylabel = []    # labels for train sentences
#_________________________________________________________________________________
def train_test_seperator(data_path, label_path, train_rate = 0.6, validate_rate = 0.1):
    data_file = open(data_path, "r")
    label_file = open(label_path, "r")

    tmp_data = data_file.readlines()
    data_content = []
    for s in tmp_data:
        s = s.split("\n")
        s = s[0]
        s = s.split("\r")
        s = s[0]
        if(s == "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"):
            continue
        else:
            data_content.append(s)

    tmp_data = label_file.readlines()
    label_content = []
    for s in tmp_data:
        s = s.split("\n")
        s = s[0]
        s = s.split("\r")
        s = s[0]
        label_content.append(int(s))

    data_file.close()
    label_file.close()

    Num_Of_train = int(len(label_content) * train_rate)
    Num_Of_validate = int(len(label_content) * validate_rate)

    train_data = []
    train_label = []

    validate_data = []
    validate_label = []

    test_data = []
    test_label = []

    permutation = np.random.permutation(len(label_content))
    index = 0
    for i in permutation:
        if index < Num_Of_train:
            train_data.append(data_content[i])
            train_label.append(label_content[i])
        elif index < Num_Of_train + Num_Of_validate:
            validate_data.append(data_content[i])
            validate_label.append(label_content[i])
        else:
            test_data.append(data_content[i])
            test_label.append(label_content[i])
        index += 1

    return(train_data, train_label, validate_data, validate_label, test_data, test_label)
############################################ End of Functions ######################################
data_path = '/media/bero/9214EFDB14EFBFF9/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/SentiPersV1.0/separated/beroData.txt'
label_path = '/media/bero/9214EFDB14EFBFF9/Users/BeRo/Google Drive/Bero/arshad project/Ms_Thesis/data set/SentiPersV1.0/separated/beroLabel.txt'

total_data = train_test_seperator(data_path = data_path, label_path= label_path, train_rate= train_rate, validate_rate = validate_rate)
train_data = total_data[0]
train_label = total_data[1]
validate_data = total_data[2]
validate_label = total_data[3]
test_data = total_data[4]
test_label = total_data[5]

index  = 0
for line in train_data:
    tmp = normalizer.normalize(line)
    #print(tmp)
    #print(sent_tokenize(tmp))
    word_tokenized = word_tokenize(tmp)
    #print(word_tokenized)
    labeledSent = TaggedDocument(words = word_tokenized, tags = [index])
    sentences.append(labeledSent)
    index += 1

num_features = 100
min_word_count = 5
context = 8
num_workers = 4
print("Training model...")
model = Doc2Vec(sentences, workers=num_workers, size = num_features, min_count = min_word_count, window = context)
print("model Trained.")

for epoch in range(num_Of_epoch):
    model.train(sentences)


mydocvec = model.docvecs
classifier = LogisticRegression()
classifier.fit(mydocvec, train_label)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)


#########################################################################################################
#                   testing
test_sentences_vec = []
index = 0
for line in test_data:
    #tmp = line.split('\n')
    #tmp = tmp[0]
    tmp = normalizer.normalize(line)
    #print(tmp)
    #print(sent_tokenize(tmp))
    word_tokenized = word_tokenize(tmp)
    #print(word_tokenized)
    myvec = model.infer_vector(word_tokenized, alpha=0.1, min_alpha=0.0001, steps=5)
    test_sentences_vec.append(myvec)
    index += 1

print(classifier.score(test_sentences_vec, test_label))

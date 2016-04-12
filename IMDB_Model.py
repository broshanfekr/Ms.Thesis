import nltk
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sknn.mlp import Classifier, Layer
import numpy as np
from bs4 import BeautifulSoup
import re
import sys
import random
import copy

################## define variables-----------------------------------------------
################################# Model parameters ###############################
Feature_Dimention=400
Learning_Rate = 0.025
Context_Size = 8
Min_Count = 1
Sample_Frequency = 1e-5
Worker_Count = 1
Training_Algorithm = 1# 1 is for pv-dm and 0 is for pv-DBOW
isHirarcal_Sampling = 1# if set to one Hirarcal Sampling is used else not
Negative_Sampling_Count = 5
Dbow_Words = 1 # if set to one traines the model with skip-gram and DBOW simultancly
Dm_mean = 0    #if set to zero uses the sum of the word vector else if set to one uses the average of word vectors
Dm_Concat = 1  #if set to one uses the concatination of context word vectors else if set to one uses the average
##########################################################################################
num_Of_epoch = 20
sentences = []  # Initialize an empty list of sentences
mylabel = []    # labels for train sentences
test_sentences = []
test_labels = []
pos_label = 1
neg_label = 0
unsup_label = -1
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#_________________________________________________________________________________
################## define functions-----------------------------------------------
def Load_Data():
    train_pos_path = '/home/bero/Desktop/dataset/aclImdb/totalforlaptop/train/IMDB_dataset_pos.txt'
    train_neg_path = '/home/bero/Desktop/dataset/aclImdb/totalforlaptop/train/IMDB_dataset_neg.txt'
    train_unsup_path = '/home/bero/Desktop/dataset/aclImdb/totalforlaptop/train/IMDB_dataset_unsup.txt'
    test_pos_path = '/home/bero/Desktop/dataset/aclImdb/totalforlaptop/test/IMDB_dataset_pos.txt'
    test_neg_path = '/home/bero/Desktop/dataset/aclImdb/totalforlaptop/test/IMDB_dataset_neg.txt'

    Read_From_File(train_pos_path, pos_label, sentences, mylabel)
    print("posetive data parsed from training set")
    Read_From_File(train_neg_path, neg_label, sentences, mylabel)
    print("negative data parsed from training set")
    Read_From_File(train_unsup_path, unsup_label, sentences, mylabel)
    print("unsup data parsed from training set")

    Read_From_File(test_pos_path, pos_label, test_sentences, test_labels)
    Read_From_File(test_neg_path, neg_label, test_sentences, test_labels)
    print("sentences parsed from testing set")

def Read_From_File(file_path, label, sentences, mylabel):
    data_file = open(file_path, 'r')
    data = data_file.readlines()
    Num_Of_Samples = len(data)
    offset = len(sentences)
    for i in range(Num_Of_Samples):#train["review"]:
        #print(i)
        review = data[i]
        mylabel.append(label)
        sentences.append(Review2Doc(review = review, mytag=i+offset))
    data_file.close()

def Review2Doc(review, mytag, remove_stopwords=False):
    #    review_text = BeautifulSoup(review).get_text()    #Remove HTML tags
    review_text = re.sub("[^a-zA-Z]", " ", review)  # Remove non-letters from words
    words = review_text.lower().split()  # Convert words to lower case and split them
    for i in range(len(words)):
        words[i] = unicode(words[i])
    # remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    labeledSent = TaggedDocument(words = words, tags = [mytag])
    return labeledSent

def Train_Model(Epoch_Number, model, Training_data, isSave=True, Save_Frequency = 10):
    for epoch in range(Epoch_Number):
        print("epoch Number: ", epoch)
        if (isSave and epoch % Save_Frequency == 0 and epoch!=0):
            #modelname = './myIMDB_model.d2v' + str(epoch)
            modelname = './myIMDB_model.d2v'
            model.save(modelname)
            print("model saved successfully")
        random.shuffle(Training_data)
        model.train(Training_data)
    model.save('./myIMDB_model.d2v')
    print("model Trained successfully")

def Load_Model(name='./myIMDB_model.d2v'):
    return Doc2Vec.load(name)

def train_Classifier(Doc_Vector):
    mydocvec = []
    docvec_labels = []
    for i in range(len(Doc_Vector)):
        if (mylabel[i] != -1):
            mydocvec.append(Doc_Vector[i])
            docvec_labels.append(mylabel[i])

    nn = Classifier(layers=[Layer("Rectifier", units=100), Layer("Softmax")],
                    learning_rate=0.02,batch_size=1,n_iter=500)
    nn.fit(np.asarray(mydocvec), np.asarray(docvec_labels))

    classifier = LogisticRegression()
    classifier.fit(mydocvec, docvec_labels)
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

    return(nn, classifier)

def test_Classifier(classifier, model, extra_model = None):
    test_sentences_vec = []
    if(extra_model == None):
        for sent in test_sentences:
            mydocwords = sent[0]
            myvec = model.infer_vector(mydocwords, alpha=0.1, min_alpha=0.0001, steps=50)
            test_sentences_vec.append(myvec)
    else:
        for sent in test_sentences:
            mydocwords = sent[0]
            first_vec = np.asarray(model.infer_vector(mydocwords, alpha=0.1, min_alpha= 0.0001, steps=50))
            sec_vec = np.asarray(extra_model.infer_vector(mydocwords, alpha=0.1, min_alpha= 0.0001, steps=50))
            test_sentences_vec.append(np.concatenate((first_vec, sec_vec), axis=1))

    score = classifier.score(np.asarray(test_sentences_vec), np.asarray(test_labels))
    return score

def main(argv):
    print(argv)

    Load_Data()

    #PV_DM_Model = Doc2Vec(size=Feature_Dimention, alpha=Learning_Rate, window=Context_Size, min_count=Min_Count,
    #                     sample=Sample_Frequency, workers=Worker_Count, dm=Training_Algorithm, hs=isHirarcal_Sampling,
    #                     negative=Negative_Sampling_Count, dbow_words=Dbow_Words, dm_mean=Dm_mean,
    #                     dm_concat=Dm_Concat)
    PV_DM_Model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=1)
    print("PV_DM_Model created successfully")
    Training_Algorithm = 0 # to build a pv_dbow model
    #PV_DBOW_Model = Doc2Vec(size=Feature_Dimention, alpha=Learning_Rate, window=Context_Size, min_count=Min_Count,
    #                        sample=Sample_Frequency, workers=Worker_Count, dm=Training_Algorithm, hs=isHirarcal_Sampling,
    #                        negative=Negative_Sampling_Count, dbow_words=Dbow_Words, dm_mean=Dm_mean, dm_concat=Dm_Concat)
    PV_DBOW_Model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, dm = 0, negative=5, workers=1)
    print("PV_DBOW_Model created successfully")

    permuted_sentences = copy.deepcopy(sentences)
    random.shuffle(permuted_sentences)

    PV_DM_Model.build_vocab(sentences)
    PV_DBOW_Model.build_vocab(sentences)

    print("Training PV_DM_Model...")
    Train_Model(Epoch_Number=num_Of_epoch, model=PV_DM_Model, Training_data=permuted_sentences,
                isSave=True, Save_Frequency=10)

    print("Training PV_DBOW Model...")
    Train_Model(Epoch_Number=num_Of_epoch, model=PV_DBOW_Model, Training_data=permuted_sentences,
                isSave=True, Save_Frequency=10)


    pvdm_vectors = np.asarray(PV_DM_Model.docvecs)
    pvdbow_vectors = np.asarray(PV_DBOW_Model.docvecs)
    concat_vector = np.concatenate((pvdm_vectors, pvdbow_vectors), axis=1)

    nn, classifier = train_Classifier(concat_vector)

    score = test_Classifier(classifier, PV_DM_Model , PV_DBOW_Model)
    print("Logistic Regression score is: ", score)

    score = test_Classifier(nn, PV_DM_Model, PV_DBOW_Model)
    print("NN score is: ", score)


if __name__ == "__main__":
    main(sys.argv)
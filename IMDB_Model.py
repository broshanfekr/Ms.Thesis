import nltk
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
import re

################## define variables-----------------------------------------------
num_features=100
num_Of_epoch = 50
sentences = []  # Initialize an empty list of sentences
mylabel = []    # labels for train sentences
pos_label = 1
neg_label = 0
unsup_label = -1
#_________________________________________________________________________________
################## define functions-----------------------------------------------
def Review2Word(review, remove_stopwords=False):
#    review_text = BeautifulSoup(review).get_text()    #Remove HTML tags
    review_text = re.sub("[^a-zA-Z]"," ", review)   #Remove non-letters from words
    words = review_text.lower().split()     #Convert words to lower case and split them
    #remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)   #Return a list of words

def Review2Doc(review, mytag, remove_stopwords=False):
    labeledSent = TaggedDocument(words = Review2Word(review, remove_stopwords), tags = [mytag])
    return labeledSent

def Read_From_File(file_path, label, sentences, mylabel):
    data_file = open(file_path, 'r')
    data = data_file.readlines()
    Num_Of_Samples = len(data)
    offset = len(sentences)
    for i in range(Num_Of_Samples):#train["review"]:
        #print(i)
        review = data[i]
        mylabel.append(label)
        sentences.append(Review2Doc(review = review, mytag = i+offset))
    data_file.close()
############################### End Functions ###################################################
# Read data from files
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#train_pos_path = '/home/bero/Desktop/dataset/aclImdb/total/train/IMDB_dataset_pos.txt'
#train_neg_path = '/home/bero/Desktop/dataset/aclImdb/total/train/IMDB_dataset_neg.txt'
#train_unsup_path = '/home/bero/Desktop/dataset/aclImdb/total/train/IMDB_dataset_unsup.txt'
train_pos_path = '/home/bero/Desktop/codes/IMDB/total/train/IMDB_dataset_pos.txt'
train_neg_path = '/home/bero/Desktop/codes/IMDB/total/train/IMDB_dataset_neg.txt'
train_unsup_path = '/home/bero/Desktop/codes/IMDB/total/train/IMDB_dataset_unsup.txt'


Read_From_File(train_pos_path, pos_label, sentences, mylabel)
print("posetive data parsed from training set")
Read_From_File(train_neg_path, neg_label, sentences, mylabel)
print("negative data parsed from training set")
Read_From_File(train_unsup_path, unsup_label, sentences, mylabel)
print("unsup data parsed from training set")
############################### End of Reading ###################################################
# creating the model
num_features = 400
min_word_count = 3
context = 8
num_workers = 3
model = Doc2Vec(documents=sentences, workers=num_workers, size=num_features, min_count = min_word_count, window=context, dm=1, dm_concat=1, dbow_words=1)
#model = Doc2Vec(sentences, workers=num_workers, size = num_features, min_count = min_word_count, window = context)
print("model created succssfully")

 # training the model/home/bero/Desktop/dataset/aclImdb/totalforlaptop
for epoch in range(num_Of_epoch):
    print("epoch Number: ", epoch)
    if(epoch%10 == 0):
        modelname = './myIMDB_model.d2v' + str(epoch)
        model.save(modelname)
    model.train(sentences)
print("model Trained successfully")

model.save('./myIMDB_model.d2v')
############################### End of Model Creating ###################################################
# using model for sentiment analysis
# training a logisticRegression for sentiment analysing
model = Doc2Vec.load('./myIMDB_model.d2v')

tmpdocvec = model.docvecs
mydocvec = []
docvec_labels = []
for i in range(len(tmpdocvec)):
    if(mylabel[i] != -1):
        mydocvec.append(tmpdocvec[i])
        docvec_labels.append(mylabel[i])

classifier = LogisticRegression()
classifier.fit(mydocvec, docvec_labels)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
########################### End Of Training Logistic Regression model #####################################
#                   testing
#classifier.score(test_arrays, test_labels)
# Reading from data
#test_pos_path = '/home/bero/Desktop/dataset/aclImdb/total/train/IMDB_dataset_pos.txt'
#test_neg_path = '/home/bero/Desktop/dataset/aclImdb/total/train/IMDB_dataset_neg.txt'
test_pos_path = '/home/bero/Desktop/codes/IMDB/total/test/IMDB_dataset_pos.txt'
test_neg_path = '/home/bero/Desktop/codes/IMDB/total/test/IMDB_dataset_neg.txt'

test_sentences = []
test_labels = []
Read_From_File(test_pos_path, pos_label, test_sentences, test_labels)
Read_From_File(test_neg_path, neg_label, test_sentences, test_labels)
print("sentences parsed from testing set")

test_sentences_vec = []
Num_Of_Test_Samples = len(test_sentences)
for sent in test_sentences:
    mydocwords = sent[0]
    myvec = model.infer_vector(mydocwords, alpha=0.1, min_alpha=0.0001, steps=5)
    test_sentences_vec.append(myvec)

print("the accuracy is: ")
print(classifier.score(test_sentences_vec, test_labels))

from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sknn.mlp import Classifier, Layer
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
import re
import sys
import tarfile
import random
import copy
import logging, sys, pprint
import timeit

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
################## define variables-----------------------------------------------
num_Of_epoch = 25
is_remove_stopwords = False
sentences = []  # Initialize an empty list of sentences
mylabel = []    # labels for train sentences
test_sentences = []
test_labels = []
pos_label = 1
neg_label = 0
unsup_label = -1
dataset_file_name = "total.tar"
destfile = open("Sentiment_result", "w")
#_________________________________________________________________________________
################## define functions-----------------------------------------------
def Load_Data(is_remove_stopwords):
    tar = tarfile.open(dataset_file_name)
    for member in tar.getmembers():
        file_name = member.name.split("_")
        isTrain = file_name[0]
        data_label = file_name[-1].split(".")[0]
        f = tar.extractfile(member)
        content = f.readlines()
        f.close()
        if(isTrain == "train"):
            if(data_label == "pos"):
                Read_From_File(content, pos_label, sentences, mylabel, is_remove_stopwords)
                logging.info('posetive data parsed from training set')
            elif(data_label == "neg"):
                Read_From_File(content, neg_label, sentences, mylabel, is_remove_stopwords)
                logging.info("negative data parsed from training set")
            else:
                Read_From_File(content, unsup_label, sentences, mylabel, is_remove_stopwords)
                logging.info("unsup data parsed from training set")
        elif(isTrain == "test"):
            if(data_label == "pos"):
                Read_From_File(content, pos_label, test_sentences, test_labels, is_remove_stopwords)
                logging.info("posetive sentences parsed from testing set")
            elif(data_label == "neg"):
                Read_From_File(content, neg_label, test_sentences, test_labels, is_remove_stopwords)
                logging.info("negative sentences parsed from testing set")
    tar.close()

def Read_From_File(data, label, sentences, mylabel, is_remove_stopwords):
    Num_Of_Samples = len(data)
    offset = len(sentences)
    for i in range(Num_Of_Samples):#train["review"]:
        #print(i)
        review = data[i]
        mylabel.append(label)
        sentences.append(Review2Doc(review = review, mytag=i+offset, remove_stopwords=is_remove_stopwords))

def Review2Doc(review, mytag, remove_stopwords=False):
    review_text = review
    review_text = BeautifulSoup(review_text, "lxml").get_text() #Remove HTML tags
    #words = word_tokenize(review_text)
    words = wordpunct_tokenize(review_text)

    #review_text = re.sub("[^a-zA-Z]", " ", review_text)  # Remove non-letters from words
    #words = review_text.lower().split()  # Convert words to lower case and split them
    for i in range(len(words)):
        words[i] = words[i].lower()
        words[i] = unicode(words[i])
    # remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    labeledSent = TaggedDocument(words=words, tags=[mytag])
    return labeledSent

def Train_Model(Epoch_Number, model, model_name, Training_data, isSave=True, Save_Frequency=10):
    for epoch in range(Epoch_Number):
        logging.info("epoch Number: " + str(epoch))
        if (isSave and epoch % Save_Frequency == 0 and epoch!=0):
            modelname = './myIMDB_model_' + model_name + '.d2v'
            model.save(modelname)
            logging.info("model saved successfully")
        #random.shuffle(Training_data)
        model.train(Training_data)
        model.alpha *= 0.99
        model.min_alpha = model.alpha
        if(model.alpha < 1e-4):
            break
    model.save('./myIMDB_model_' + model_name + '.d2v')
    logging.info("model Trained successfully")

def Load_Model(name='./myIMDB_model.d2v'):
    return Doc2Vec.load(name)

def train_Classifier(Doc_Vector):
    mydocvec = []
    docvec_labels = []
    for i in range(len(Doc_Vector)):
        if (mylabel[i] != -1):
            mydocvec.append(Doc_Vector[i])
            docvec_labels.append(mylabel[i])


    logging.info("training logistic regression classifier...")
    classifier = LogisticRegression()
    classifier.fit(mydocvec, docvec_labels)

    logging.info("training logistic Neural network classifier")
    nn = Classifier(layers=[Layer("Rectifier", units=100), Layer("Softmax")],
                    learning_rate=0.02,batch_size=1,n_iter=100)
    nn.fit(np.asarray(mydocvec), np.asarray(docvec_labels))

    return(nn, classifier)

def test_Classifier(classifier, model, extra_model = None):
    test_sentences_vec = []
    if(extra_model == None):
        for sent in test_sentences:
            mydocwords = sent[0]
            myvec = model.infer_vector(mydocwords, alpha=0.05, min_alpha=0.0001, steps=50)
            test_sentences_vec.append(myvec)
    else:
        for sent in test_sentences:
            mydocwords = sent[0]
            first_vec = np.asarray(model.infer_vector(mydocwords, alpha=0.05, min_alpha= 0.0001, steps=50))
            sec_vec = np.asarray(extra_model.infer_vector(mydocwords, alpha=0.1, min_alpha= 0.0001, steps=50))
            test_sentences_vec.append(Concat_Paragraph_Vector(v1=first_vec, v2= sec_vec))

    score = classifier.score(np.asarray(test_sentences_vec), np.asarray(test_labels))
    return score

def Concat_Paragraph_Vector(v1, v2):
    result = np.hstack((v1, v2))
    #result = v1 + v2
    return result

def main(argv):
    print(argv)
    ################################# Model parameters ###############################
    Feature_Dimention = 150
    Learning_Rate = 0.05
    Context_Size = 10
    Min_Count = 2
    Sample_Frequency = 1e-2
    Worker_Count = 8
    Training_Algorithm = 1  # 1 is for pv-dm and 0 is for pv-DBOW
    isHirarcal_Sampling = 1  # if set to one Hirarcal Sampling is used else not
    Negative_Sampling_Count = 25
    Dbow_Words = 0  # if set to one traines the model with skip-gram and DBOW simultancly
    Dm_mean = 0  # if set to zero uses the sum of the word vector else if set to one uses the average of word vectors
    Dm_Concat = 1  # if set to one uses the concatination of context word vectors else if set to one uses the average
    ##########################################################################################
    Load_Data(is_remove_stopwords)

    PV_DM_Model = Doc2Vec(documents=sentences,size=Feature_Dimention, alpha=Learning_Rate, min_alpha=Learning_Rate, window=Context_Size,
                          min_count=Min_Count, sample=Sample_Frequency, workers=Worker_Count, dm=Training_Algorithm,
                          hs=isHirarcal_Sampling, negative=Negative_Sampling_Count, dbow_words=Dbow_Words,
                          dm_mean=Dm_mean, dm_concat=Dm_Concat)
    #PV_DM_Model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=1)
    logging.info("PV_DM_Model created successfully")
    Training_Algorithm = 0 # to build a pv_dbow model
    Sample_Frequency = 1e-4
    PV_DBOW_Model = Doc2Vec(documents=sentences, size=Feature_Dimention, alpha=Learning_Rate, min_alpha=Learning_Rate, window=Context_Size,
                            min_count=Min_Count, sample=Sample_Frequency, workers=Worker_Count, dm=Training_Algorithm,
                            hs=isHirarcal_Sampling, negative=Negative_Sampling_Count, dbow_words=Dbow_Words,
                            dm_mean=Dm_mean, dm_concat=Dm_Concat)
    #PV_DBOW_Model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, dm = 0, negative=5, workers=1)
    logging.info("PV_DBOW_Model created successfully")

    #permuted_sentences = copy.deepcopy(sentences)
    #random.shuffle(permuted_sentences)

    #PV_DM_Model.build_vocab(sentences)
    #PV_DBOW_Model.build_vocab(sentences)

    logging.info("Training PV_DM_Model...")
    Train_Model(Epoch_Number=num_Of_epoch, model=PV_DM_Model, model_name='PV_DM', Training_data=sentences,
                isSave=True, Save_Frequency=20)

    logging.info("Training PV_DBOW Model...")
    Train_Model(Epoch_Number=num_Of_epoch, model=PV_DBOW_Model, model_name='PV_DBOW', Training_data=sentences,
                isSave=True, Save_Frequency=40)


    pvdm_vectors = np.asarray(PV_DM_Model.docvecs)
    pvdbow_vectors = np.asarray(PV_DBOW_Model.docvecs)
    concat_vector = Concat_Paragraph_Vector(v1=pvdm_vectors, v2=pvdbow_vectors)
    ############################################################################################
    #import matplotlib.pyplot as plt
    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=2)
    #X_r = pca.fit_transform(concat_vector)
    #for i in range(len(X_r)):
    #    if(mylabel[i] == 0):
    #        color = 'r'
    #    if(mylabel[i] == 1):
    #        color = 'b'
    #    plt.scatter(X_r[i,0], X_r[i,1], color= color )
    #plt.show()
    #from sklearn.manifold import TSNE
    #import matplotlib.pyplot as plt

    #ts = TSNE(2)
    #reduced_vecs = ts.fit_transform(concat_vector)
    #for i in range(len(reduced_vecs)):
    #    if(mylabel[i] == 0):
    #        color = 'r'
    #    if(mylabel[i] == 1):
    #        color = 'b'
    #    plt.scatter(reduced_vecs[i,0], reduced_vecs[i,1], color= color )
    #plt.show()
    #############################################################################################
    nn, classifier = train_Classifier(concat_vector)

    score = test_Classifier(classifier, PV_DM_Model , PV_DBOW_Model)
    destfile.write("Logistic Regression score is: " + str(score))
    destfile.write("\n")
    print("Logistic Regression Score is: ", score)

    score = test_Classifier(nn, PV_DM_Model, PV_DBOW_Model)
    destfile.write("NN score is: " + str(score))
    destfile.write("\n")
    print("NN Score is: ", score)

if __name__ == "__main__":
    start = timeit.default_timer()
    main(sys.argv)
    stop = timeit.default_timer()
    spent_time = int(stop - start)
    sec = spent_time % 60
    spent_time = spent_time / 60
    minute = spent_time % 60
    spent_time = spent_time / 60
    hours = spent_time
    logging.info("h: " + str(hours) + "  minutes: " + str(minute) + "  secunds: " + str(sec))
    destfile.write("h: " + str(hours) + "  minutes: " + str(minute) + "  secunds: " + str(sec))
    destfile.close()

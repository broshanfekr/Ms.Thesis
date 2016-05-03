from gensim.models.doc2vec import TaggedDocument
from bs4 import BeautifulSoup
from nltk.tokenize import wordpunct_tokenize
from collections import namedtuple
from sklearn.linear_model import LogisticRegression
import tarfile
import logging, sys
from gensim.models import Doc2Vec
from collections import OrderedDict
import multiprocessing
import numpy as np
from random import sample
from collections import defaultdict
from random import shuffle
import datetime

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
dataset_file_name = "total.tar"

########################################## Define Functions##############################################
best_error = defaultdict(lambda :1.0)  # to selectively-print only best errors achieved

def Load_Data():
    train_pos = []
    train_neg = []
    train_unsup = []
    test_pos = []
    test_neg = []
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
                train_pos = content
                logging.info('posetive data parsed from training set')
            elif(data_label == "neg"):
                train_neg = content
                logging.info("negative data parsed from training set")
            else:
                train_unsup = content
                logging.info("unsup data parsed from training set")
        elif(isTrain == "test"):
            if(data_label == "pos"):
                test_pos = content
                logging.info("posetive sentences parsed from testing set")
            elif(data_label == "neg"):
                test_neg = content
                logging.info("negative sentences parsed from testing set")
    alldocs = train_pos + train_neg + train_unsup + test_pos + test_neg
    tar.close()
    return alldocs

def logistic_predictor_from_data(train_targets, train_regressors):
    classifier = LogisticRegression()
    classifier.fit(np.asarray(train_regressors), np.asarray(train_targets))
    return classifier

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""
    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    #train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    #test_regressors = sm.add_constant(test_regressors)

    # predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)
##################################################################################################################

alldocs = []# to hold all reviews in original order
alldata = Load_Data()
for line_no, line in enumerate(alldata):
    if line_no == 1000:
        break
    review_text = BeautifulSoup(line, "lxml").get_text()  # Remove HTML tags
    words = wordpunct_tokenize(review_text)
    for i in range(len(words)):
        words[i] = words[i].lower()
        words[i] = unicode(words[i])
    #tokens = gensim.utils.to_unicode(line).split()
    tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
    split = ['train','test','extra','extra'][line_no//250]  # 25k train, 25k test, 25k extra
    sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//125] # [12.5K pos, 12.5K neg]*2 then unknown
    alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = train_docs[:]  # for reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))
##############################################################################################
cores = multiprocessing.cpu_count()
simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=150, window=10,sample=1e-2, negative=25, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=150, negative=25,window=10, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=150, window=10, negative=25, hs=0, min_count=2, workers=cores),
]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)
#########################################################################################################
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])
##########################################################################################################

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes
print("START %s" % datetime.datetime.now())

for epoch in range(passes):
    shuffle(doc_list)  # shuffling gets best results
    for name, train_model in models_by_name.items():
        # train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        train_model.train(doc_list)

        # evaluate
        if ((epoch + 1) % 5) == 0 or epoch == 0:
            infer_err, err_count, test_count, predictor = error_rate_for_model(train_model, train_docs, test_docs, infer=True, infer_subsample=1)
            best_indicator = ' '
            if infer_err < best_error[name + '_inferred']:
                best_error[name + '_inferred'] = infer_err
                best_indicator = '*'
            print("%s%f : %i passes : %s" % (best_indicator, infer_err, epoch + 1, name + '_inferred'))

    print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))

###############################################################################################
# print best error rates achieved
for rate, name in sorted((rate, name) for name, rate in best_error.items()):
    print("%f %s" % (rate, name))

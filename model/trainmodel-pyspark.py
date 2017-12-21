import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark import SparkConf
import csv
from itertools import izip
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import SVM
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

with open('/Users/shuyang/Documents/bigdata/Sentence-Extraction-Based-Article-Summarization/train_data.txt', 'rb') as f:
    trainDATA = pickle.load(f)


with open('/Users/shuyang/Documents/bigdata/Sentence-Extraction-Based-Article-Summarization/train_label.txt', 'rb') as f:
    trainLABEL = pickle.load(f)


best_score=[]
test_score=[]
train_data=[]
train_label=[]
test_real_data=[]
test_real_label=[]
for data in trainDATA:
    train_data.extend(data)
for data in trainLABEL:
    train_label.extend(data)

train_data = train_data[:len(train_label)]


#train-test splitting: 60% train data, 40% test data, stratified sampling
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.4, random_state=0)

rms.fit(X_train, y_train)
a = rms.predict(X_test)
rms.score(X_test,y_test)

np.count_nonzero(y_test)
np.count_nonzero(a)

SVM linear kenerl: best c=0.1, score=0.59;test score=0.59
param_grid = [
 {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']},
]

clf = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy')
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_estimator_)
#print(clf.grid_scores_)
print(clf.best_score_)
best_score.append(clf.best_score_)

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

c=0.1
svc1=  SVC(kernel='linear', C=c)
clf1 = svc1.fit(X_train, y_train)
score = clf1.score(X_test, y_test)
print "best test score for linear kernel is : %s\n" % score
#test_score.append(score)

#clf1.predict(test_real_data)

# Train a naive Bayes model.
clf2 = NaiveBayes.train(training, modelType = "multinomial")
predictionAndLabel = test.map(lambda p: (clf2.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda x, v: x == v).count() / test.count()
print accuracy

# Train a navie Bayes model (Bernoulli)
clf3 = NaiveBayes.train(train_data, modelType = "Bernoulli")
predictionAndLabel = test.map(lambda p: (clf2.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda x, v: x == v).count() / test.count()
print accuracy


import cPickle
# save the classifier
with open('NB_big_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf2, fid)

# load it again
with open('NB_big_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)

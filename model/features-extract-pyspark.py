import re
import nltk
#import nltk.data
#from nltk.corpus import stopwords
rom pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Normalizer
from pyspark.ml.clustering import KMeans
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark import SparkConf
import numpy as np
import pickle
import sys
import os


#nltk.download("punkt")
#stop_words = set(stopwords.words('english'))

stop_words = {u'a',
 u'about',
 u'above',
 u'after',
 u'again',
 u'against',
 u'ain',
 u'all',
 u'am',
 u'an',
 u'and',
 u'any',
 u'are',
 u'aren',
 u'as',
 u'at',
 u'be',
 u'because',
 u'been',
 u'before',
 u'being',
 u'below',
 u'between',
 u'both',
 u'but',
 u'by',
 u'can',
 u'couldn',
 u'd',
 u'did',
 u'didn',
 u'do',
 u'does',
 u'doesn',
 u'doing',
 u'don',
 u'down',
 u'during',
 u'each',
 u'few',
 u'for',
 u'from',
 u'further',
 u'had',
 u'hadn',
 u'has',
 u'hasn',
 u'have',
 u'haven',
 u'having',
 u'he',
 u'her',
 u'here',
 u'hers',
 u'herself',
 u'him',
 u'himself',
 u'his',
 u'how',
 u'i',
 u'if',
 u'in',
 u'into',
 u'is',
 u'isn',
 u'it',
 u'its',
 u'itself',
 u'just',
 u'll',
 u'm',
 u'ma',
 u'me',
 u'mightn',
 u'more',
 u'most',
 u'mustn',
 u'my',
 u'myself',
 u'needn',
 u'no',
 u'nor',
 u'not',
 u'now',
 u'o',
 u'of',
 u'off',
 u'on',
 u'once',
 u'only',
 u'or',
 u'other',
 u'our',
 u'ours',
 u'ourselves',
 u'out',
 u'over',
 u'own',
 u're',
 u's',
 u'same',
 u'shan',
 u'she',
 u'should',
 u'shouldn',
 u'so',
 u'some',
 u'such',
 u't',
 u'than',
 u'that',
 u'the',
 u'their',
 u'theirs',
 u'them',
 u'themselves',
 u'then',
 u'there',
 u'these',
 u'they',
 u'this',
 u'those',
 u'through',
 u'to',
 u'too',
 u'under',
 u'until',
 u'up',
 u've',
 u'very',
 u'was',
 u'wasn',
 u'we',
 u'were',
 u'weren',
 u'what',
 u'when',
 u'where',
 u'which',
 u'while',
 u'who',
 u'whom',
 u'why',
 u'will',
 u'with',
 u'won',
 u'wouldn',
 u'y',
 u'you',
 u'your',
 u'yours',
 u'yourself',
 u'yourselves'}

def title_stem(title):
    porter_stemmer = PorterStemmer()
    title_stem_list = []
    for word in title.split(" "):
        if word not in stop_words:
            word=re.sub(r'[^\w]','', word)
            title_stem_list.append(porter_stemmer.stem(word.lower()))
    return title_stem_list

def count_title(s, title):
    title_stem_list = title_stem(title)
#    print title_stem_list
    count = 0
    for word in s:
        if word in title_stem_list:
            count += 1
    return count

def count_important_words(s, top):
    count = 0
    for word in s:
        if word in top:
            count += 1
    return count



def create_training_data(file_name, title):

    sentences_arr = []
    para_arr = []

    paragraphs = file_name.splitlines()

    #print paragraphs

    for paragraph in paragraphs:
        if paragraph.isspace():
            continue
        sentences_arr = []
        para = nltk.sent_tokenize(paragraph)
        if para != []:
            for sentences in para:
                sentences_arr.append(sentences.split())
        if sentences_arr != []:
            para_arr.append(sentences_arr)

    stem_txt_final=[]
    porter_stemmer = PorterStemmer()

    for p in para_arr:
        stem_txt = []
        for s in p:
            temp=[]
            for word in s:
                if word not in stop_words:
                    word=re.sub(r'[^\w]','', word)
                    temp.append(porter_stemmer.stem(word.lower()))
            stem_txt.append(temp)
        stem_txt_final.append(stem_txt)


#    title_list = stem_txt_final[0][0][0]


    words = []
    for para in stem_txt_final:
        article_string = ['  '.join(mylist) for mylist in para]
        words.extend(article_string)
    
    
    #tokenize the document text
    tokenizer = Tokenizer(inputCol="article", outputCol="tokens")
    words = tokenizer.transform(words).cache()

    hashingTF = HashingTF (inputCol="stopWordsRemovedTokens", outputCol="rawFeatures", numFeatures=2000)
    words = hashingTF.transform(words).cache()
    words = words.drop('stopWordsRemovedTokens') 
    words = words.drop('tokens') 


    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
    idfModel = idf.fit(words)
    df_article = idfModel.transform(words).cache()

    ################################################################
    top_n = 10
    top_features = [features[i] for i in indices[:top_n]]
    #print top_features

    count = 0
    features = []
    for para in stem_txt_final:
        position = 0
        for s in para:
            position += 1
            count += 1
            feat = [position, count, count_title(s, title), count_important_words(s, top_features), len(s)]
            features.append(feat)
    return features




d = "/Users/shuyang/Downloads/test_model/docs/"
h = "/Users/shuyang/Downloads/test_model/highlights/"

features = []
import io
def run():
    count = 0
    for file in os.listdir("/Users/shuyang/Downloads/test_model/docs"):
        
        
        temp_name = os.path.join("", file)
        a = temp_name.split('_')

        name_hlights = h + a[0] + '_' + 'highlight' + '_' + a[-1]
        name = d + os.path.join("", file)
        with io.open(name_hlights, encoding='ISO-8859-1') as f:
            print name_hlights
            title = f.read()
#            print "title:",type(title)
        with io.open(name, encoding='ISO-8859-1') as f:
            print temp_name
            read_data1 = f.read()
#            print "data:",type(read_data1)
            if len(read_data1)==0:
                name_delete.append(temp_name)
                print "continue"
                continue
        try:
            count += 1
            print count
#            read_data = read_data1.decode("utf8")
            feature = create_training_data(read_data1, title)
            features.append(feature)
#            print features
        except:
            print "pass"
            name_delete.append(temp_name)
            pass

                        #pickle.dump(feature, out_file)
                        #f.close()


name_delete=[]
run()

out_file = open('train_data.txt', 'wb')
pickle.dump(features, out_file)
#print len(features[0])
out_file.close()

name_delete_file = open('name_delete.txt', 'wb')

pickle.dump(name_delete, name_delete_file)
name_delete_file.close()
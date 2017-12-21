import nltk
from rouge import Rouge
import pickle
import os
# nltk.download()
#from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
import io

rouge = Rouge()
d='/Users/shuyang/Downloads/test_model/docs/'
h='/Users/shuyang/Downloads/test_model/highlights/'

with open('/Users/shuyang/Documents/bigdata/Sentence-Extraction-Based-Article-Summarization/name_delete.txt', 'rb') as f:
    name_del = pickle.load(f)
name_del.append('.DS_Store')
result=[]
name=[]
count = 0
for file in os.listdir("/Users/shuyang/Downloads/test_model/docs"):

#for file in os.listdir("/Users/shuyang/Downloads/test/docs"):
    # if file.endswith(".txt"):
    if os.path.join("", file) not in name_del:
        os_path=os.path.join("", file)
        name_docs= d + os.path.join("", file)
        a = os_path.split('_')
        name_hlights=h + a[0] + '_' + 'highlight' + '_' + a[-1]
        # name_hlights=h+os_path+'highlight'
        name.append(name_docs)
        with io.open(name_docs,encoding='ISO-8859-1') as f:
            print name_docs
            docs = f.read()
        with io.open(name_hlights,encoding='ISO-8859-1') as f:
            print name_hlights
            hlights = f.read()
        #paragraphs = docs.splitlines()
        paragraphs = docs.split('\n')
        label_list=[]
        for paragraph in paragraphs:
            if paragraph.isspace():
                continue
            para = nltk.sent_tokenize(paragraph)

            if para != []:
                for sentences in para:
                    label_list.append(sentences)

        score_result=[]
        for sentence_label in label_list:
#            try:
                scores = rouge._get_avg_scores(sentence_label, hlights)
#                print scores
#            except:
#                pass
            #print scores
#            try:
                scores=scores['rouge-1']['f']
                print "score: ",scores
#            except:
#                scores=0.0
                score_result.append(scores)

#            score_result.append(scores)
        count += 1
#        features count is :894920
        print "count:",count
        index_sorted=sorted(range(len(score_result)), key=lambda i:score_result[i])
        index_sorted.reverse()
        top_4=index_sorted[:4]
        label=np.array(score_result)
        label[top_4]=1
        other=index_sorted[4:]
        label[other]=0
        label=list(label)
        label=[int(i) for i in label]
        print "label length:", len(label)
        result.append(label)



print "saving train_label.txt"
out_file = open('train_label.txt', 'wb')
pickle.dump(result, out_file)
out_file.close()
#out_file = open('train_label.txt', 'wb')
#pickle.dump(result, out_file)
#out_file.close()
#with open('test_label.txt', 'rb') as f:
#    my_list = pickle.load(f)
#




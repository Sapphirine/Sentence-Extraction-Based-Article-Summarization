from flask import Flask
from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps
import psycopg2
from datetime import datetime
import pdb
import eventregistry
import nltk
from eventregistry import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np
from unidecode import unidecode
import cPickle
# nltk.download()

YOUR_API_KEY ='07594173-183b-4b95-990a-ac99968fa133'
er = EventRegistry(apiKey = YOUR_API_KEY)
app = Flask(__name__)
stop = set(stopwords.words('english'))
# load classifier
with open('NB_big_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)

def title_stem(title):
    porter_stemmer = PorterStemmer()
    title_stem_list = []
    for word in title.split(" "):
        if word not in stop:
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
# load trained model and get predicted features: np array: [1,0,1]
def get_predict_sentences(features):

	labeled_sentences = gnb_loaded.predict(features)
	return labeled_sentences

# given raw text, formant list of sentences (real words)
def format_sentences(raw_text):
	senteces_extracted = []
	paragraphs = raw_text.split('\n\n')
	for paragraph in paragraphs:
		if paragraph.isspace():
			continue
        para = nltk.sent_tokenize(paragraph)
        if para != []:
        	for sentence in para:
        		senteces_extracted.append(sentence)
	return senteces_extracted

# calls format_sentences
def extract_sentences(raw_text, labeled_sentences, length):
	result_string = ""
	s = "\n"
	num_importance = np.count_nonzero(labeled_sentences)
	if num_importance >= 0:
		print "indexes1:", np.where(labeled_sentences==1)[0]
		print "length", length
		ones_indexes= np.where(labeled_sentences==1)[0] #ones_indexes is np array
		ones_indexes = ones_indexes[0:length]
		print "indexes2:", ones_indexes
		senteces_extracted = format_sentences(raw_text)
    	result = [senteces_extracted[i] for i in ones_indexes]
    	result_string = s.join(result)
	return result_string


def preprocess_raw_artical(raw_text, title, title_exist = True):

    sentences_arr = []
    para_arr = []

    paragraphs = raw_text.split('\n\n')

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
                if word not in stop:
                    word=re.sub(r'[^\w]','', word)
                    temp.append(porter_stemmer.stem(word.lower()))
            stem_txt.append(temp)
        stem_txt_final.append(stem_txt)

    words = []
    for para in stem_txt_final:
        article_string = ['  '.join(mylist) for mylist in para]
        words.extend(article_string)
#    print words
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(words)
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
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
#            print s
            feat = [position, count, count_title(s, title), count_important_words(s, top_features), len(s)]
            features.append(feat)
    return features


newslist = []
data=[]


@app.route("/", methods=['GET', 'POST'])
def home():
	topnewslist =[]
	topnews = []
	active_topnews =[]
	num_topnews = 100
	query_num = 60



	q = QueryArticlesIter(lang = "eng")
	q.setRequestedResult(RequestArticlesInfo(count = num_topnews,
		returnInfo = ReturnInfo(
			articleInfo = ArticleInfoFlags(image = True, categories = True, location = True))))
	res = er.execQuery(q)
	# print res
	results = res["articles"]["results"]
	hashuri= {}
	count =0
	for x in range(num_topnews):
		result = results[x]
		if count == query_num:
			break
		if result["image"] =="" or "image" not in result.keys() \
			or result["image"] == None or not result["image"].startswith("http"):
			continue

		if count < 4:
			# print "============"
			# print result["title"]
			# print result["image"]
			# print "============"
			title = result["title"]
			body = result["body"]
			title = title.replace(u'\'', '')
			body =  body.replace(u'\'', '')
			body =  body.replace(u'\"', '')
			body = body.replace('\n', '/n')
			result["title"] = unidecode(title)
			result["body"] = body
			uri = result["uri"]
			if uri in hashuri:
				continue
			else:
				hashuri[count] = result["uri"]
				active_topnews.append(result)
				count = count + 1
		else :

			# print "============"
			# print result
			title = result["title"]
			body = result["body"]
			title = title.replace(u'\'', '')
			body =  body.replace(u'\'', '')
			body =  body.replace(u'\"', '')
			body = body.replace('\n', '/n')
			result["title"] = unidecode(title)
			result["body"] = body
			uri = result["uri"]

			if uri in hashuri:
				continue
			else:
				hashuri[count] = result["uri"]
				topnews.append(result)
			# print "============"
			if count %4 == 3:
				topnewslist.append(topnews)
				topnews =[]
			count = count + 1


	# title = active_topnews[1]["title"]
	# raw_text = active_topnews[1]["body"]



	if request.method == "POST":
		if "query" in request.form:

			print "query articles!!!!\n"
			try:
				topic=request.form['topic']
				print topic
				source =request.form['source']
				print source
				keyword =request.form['keyword']
				print keyword
				# q = QueryArticles(keywords = keyword, lang = "eng", sourceUri = er.getNewsSourceUri(source), \
				# 	categories = topic)
				# q.setRequestedResult(RequestArticlesInfo(count = 20,
				# 	returnInfo = ReturnInfo(
				# 		articleInfo = ArticleInfoFlags(image = True, categories = True, location = True))))
				# res = er.execQuery(q)
				# results = res["articles"]["results"]
				# print results

				q = QueryArticlesIter(lang = "eng", keywords = keyword, sourceUri = er.getNewsSourceUri(source))
				q.setRequestedResult(RequestArticlesInfo(count = 20,
					returnInfo = ReturnInfo(
						articleInfo = ArticleInfoFlags(image = True, categories = True, location = True))))
				res = er.execQuery(q)

				results = res["articles"]["results"]
				print results[0]


				for x in range(len(results)):
					result = results[x]
					title = result["title"]
					body = result["body"]
					title = title.replace(u'\'', '')
					# body =  body.replace(u'\'', '')
					# body =  body.replace(u'\"', '')
					# body = body.replace('\n', '/n')
					body =  body.replace(u'\ ', '\\')
					result["title"] = unidecode(title)
					result["body"] = body
					newslist.append(result)
				print "newslist len\n"
				print len(newslist)
				return render_template('index.html', newslist = newslist, \
					active_topnews = active_topnews, topnewslist = topnewslist)
			except:
				return render_template('index.html', newslist = [], \
					active_topnews = active_topnews, topnewslist = topnewslist)

		if "newsrecommend" in request.form:
			print "Recommend!!!\n"
			print request.form
			news=request.form.keys()
			print news
			print newslist
			news.remove('newsrecommend')
			news.remove('keyword')
			for ele in news:
				temp=ele.split('/')[1]
				print "==uri=="
				print temp
				for n in newslist:
					print n["uri"]
					print n["body"]
					if temp == str(n["uri"]):
						print "push one news!!!"
						data.append(n)
			print "len of data"
			print len(data)
			print "!!!!data!!!"
			print type(data)
			if len(data) > 0:
				ele= data[0]
				del data[0]
				body = ele["body"]
				title = ele["title"]
				print body
			body = body.replace('/n', '\n')

			return render_template('index.html', newslist = newslist, article = body, title = title,
				active_topnews = active_topnews, topnewslist = topnewslist)  # render a template
		if "summary" in request.form:
			print "summary!!!!\n"
			length = request.form['length']
			title = request.form['title']
			raw_text = request.form['article_text']
			try:
				length = int(length)
			except:
				# default length is 3
				length = 3
			# print "title,", title
			# print "article,", raw_text
			try:
				features = preprocess_raw_artical(raw_text, title)
				print "features:", features
				labeled_sentences = get_predict_sentences(features)
				print "labeled_sentences:", labeled_sentences
				sentences_summerized = extract_sentences(raw_text, labeled_sentences, length)
				print "!!!!sentences_summerized:", sentences_summerized
			except:
				sentences_summerized = ""


			return render_template('index.html', newslist = newslist, \
				active_topnews = active_topnews, topnewslist = topnewslist, summary_text = sentences_summerized, article = raw_text, title = title)  # render a template

		if "next" in request.form:
			if len(data) > 0:
				ele= data[0]
				del data[0]
				body = ele["body"]
				title = ele["title"]
				print body
				# body =  body.replace(u'\'', '')
				# body =  body.replace(u'\"', '')
				body = body.replace('/n', '\n')
				return render_template('index.html', newslist = newslist, article = body, title = title,
				active_topnews = active_topnews, topnewslist = topnewslist)#
			else:
				return render_template('index.html', newslist = newslist, article = "",\
				 title = "please select an article", active_topnews = active_topnews, topnewslist = topnewslist)#


	else:
		title = "Please select a article for summarization!"
		return render_template('index.html', newslist =[], active_topnews = active_topnews,\
				title = title, topnewslist = topnewslist)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
	#app.run(debug=True)

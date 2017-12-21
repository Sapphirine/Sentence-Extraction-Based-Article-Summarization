# Big Data Analystics Project
Sentence Extraction Based Text Summarization

Project ID: 201712-44

Team member:
Shuyang Zhao sz2631
Yanan Zhang yz3054
Suyang Gao sg3393

## Objective:
The news articles, nowadays, are feeding largely in each day. One website could update hundreds of articles each day.
This project aims to retrieve the most important information and summarize the content in a few lines of words.
Also, a real-time news recommendation web-application are built upon our trained model. Users can select news from CNN, BBC, New York Times, and read news in summarized format.

## Train Model
Dataset: https://github.com/philipperemy/financial-news-dataset
  1. cd model

  2. python features-extract-pyspark.py

  2. python label-extract.py

  3. python trainmodel-pyspark.py

  Pre-trained model is in /server/NB_big_classifier.pkl

## Run Application
  1. cd server

  2. python application.py

  3. go to the address using the browser
  http://localhost:5000



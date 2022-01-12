from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('omw-1.4')
# from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
import pickle

from web_scraping import scrap_article

def special_char(text):
  reviews = ''
  for x in text:
    if x.isalnum():
      reviews = reviews + x
    else:
      reviews = reviews + ' '
  return reviews

def convert_lower(text):
   return text.lower()

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
  words = word_tokenize(text)
  return [x for x in words if x not in stop_words]

# Lemmatizing the Words
def lemmatize_word(text):
  wordnet = WordNetLemmatizer()
  return " ".join([wordnet.lemmatize(word) for word in text])

app = Flask(__name__)
with open("model.pkl","rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # scrap data, preprocessing functions
    features = [x for x in request.form.values()]
    url = features[0]
    title, content = scrap_article(url)
    content = special_char(content)
    content = convert_lower(content)
    content = remove_stopwords(content)
    content = lemmatize_word(content)
    content = [content]
    prediction = model.predict(content)
    print("Prediction here: ", prediction)
    category_id_dict = {2:'national', 0:'international', 7:'industry', 3:'karnataka', 4:'tamil nadu', 6:'hyderabad', 5:'cricket', 1:'cinema'}
    prediction = category_id_dict[prediction[0]]
    # return render_template("result.html", prediction = prediction)
    return render_template('home.html',pred='Predicted label is {}'.format(prediction))

if __name__ == '__main__':
    app.run()

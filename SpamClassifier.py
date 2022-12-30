import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
spam_df=pd.read_csv("C:/Users/bhamb/OneDrive/Desktop/Work/Python/Spam Classifier/spam.csv",encoding='Windows-1252')
spam_df = spam_df.loc[:, ~spam_df.columns.str.contains('^Unnamed')]
spam_df['length']=spam_df['v2'].apply(len)
pos_df=spam_df[spam_df['v1']=='ham']
neg_df=spam_df[spam_df['v1']=='spam']
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean
spam_df_clean = spam_df['v2'].apply(message_cleaning)
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
spam_countvectorizer = vectorizer.fit_transform(spam_df['v2'])  
X=pd.DataFrame(spam_countvectorizer.toarray())
y=spam_df['v1']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print("Training.")
#NB_classifier=MultinomialNB()
#NB_classifier.fit(X_train,y_train)
#y_predict_test = NB_classifier.predict(X_test)
#print("Naive-Bayes: ")
#print(classification_report(y_test, y_predict_test))
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Gradient Boost: ")
print(classification_report(y_test, y_pred))
#model = LogisticRegression()
#model.fit(X_train, y_train)

#y_pred = model.predict(X_test)
#print("Logistic Regression: ")
#print(classification_report(y_test, y_pred))

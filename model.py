import pandas as pd
import numpy as np


data = pd.read_csv('train.csv')

print(data.head())

x=data.iloc[:,[3]]
y=data.iloc[:,[4]]



import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(x)):
    text = re.sub('[^a-zA-Z]', ' ', x['text'][i])
    text = text.lower()
    text = text.split()
    
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

spam_detect_model = MultinomialNB().fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

y_pred=pd.DataFrame(spam_detect_model.predict(X_test))



from sklearn.metrics import confusion_matrix
cm_NB=confusion_matrix(y_test,y_pred)
cm_NB



import pickle
filename = 'nlp_model.pkl'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(cv, open('tranform.pkl', 'wb'))
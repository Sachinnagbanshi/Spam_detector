import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_words(word):
    word=re.sub('[^a-z ]','',word.lower())
    x=[]
    for i in word.split():
        if i not in stopwords.words('english'):
            x.append(ps.stem(i))
    return " ".join(x)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_words(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header("It's a Spam!")
    else:
        st.header("Not a Spam.")

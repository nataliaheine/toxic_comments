import streamlit as st
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import SnowballStemmer

def start():
    # OPEN SAVED MODEL
    with open('model_toxic_comments.pkl', 'rb') as file:
        model_pipeline = pickle.load(file)

    # TOKENIZE FUNKTION
    stop_words = stopwords.words("german")
    
    def tokenized_and_stemmed(text):
        stemmer = SnowballStemmer(language="german")
        tokens = word_tokenize(text, language="german")
        tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
        tokens_without_punctuation_and_stopwords = [i for i in tokens_without_punctuation if i not in stop_words]
        stemmed_tokens = [stemmer.stem(i) for i in tokens_without_punctuation_and_stopwords]
        return stemmed_tokens
    
    # PREDICT COMMENT FUNKTION
    def predict_comment(text):
        klasse = model_pipeline.predict([text])
        proba = model_pipeline.predict_proba([text])
        if klasse == 1:
            return klasse, round(proba[0][1]*100, 2)
        else:
            return klasse, round(proba[0][0]*100, 2)
    
    st.title("Toxic comments")
    text = st.text_input("Schreibe hier den Kommentartext")
    
    if text != "":
        klasse, proba = predict_comment(text)
        st.write(f"Der Kommentar '{text}' ist **{'toxisch' if klasse==1 else 'nicht toxisch'}** mit der Wahrscheinlichkeit von **{proba}%**")

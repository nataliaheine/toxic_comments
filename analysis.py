import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import string

def start():
    st.title("Datenanalyse & Modellauswahl")
    st.info("Dataset findet man unter folgendem [Link](https://www.kaggle.com/datasets/shubheshswain/jigsaw-toxic-comment-classification-german).")

    st.header("So sehen die ursprunglichen Daten aus:")
    df = pd.read_csv("toxic_comments_de_unprocessed.csv")
    st.dataframe(df.head(10))

    st.write("Das zukünftige Modell braucht nur 1 Label statt 6. Ich habe alles zu einem Label gebracht und zwar nach folgendem Prinzip: wenn ein Kommentar zumindest bei einem der Labels mit '1' markiert wurde, bekam er bei dem neuen Label auch ein '1'.  \n\nSomit wurden auch fast harmlose Kommentare als toxisch gelabelt. Dies könnte man ändern indem man bei Erstellung des neues Labels nicht alle der alten Labels beachtet, sondern z.B. nur die Features 'severe toxic', 'threat' und 'insult'") 
    st.write("Außerdem gab es im gesamten Dataset 133.559 nicht toxische und nur 15.086 toxische Kommentare, was beim Trainieren des Modells dazu führte, dass das Modell sich dafür entschied, die meisten Kommentare als nicht toxisch zu markieren und damit 95% accuracy zu erreichen")
    
    st.header("So sah das neue DataFrame aus mit nur 1 Label und der gleichen Anzahl an Kommentaren jeder Klasse:")
    df["label"] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].any(axis=1).astype(int)
    df = df.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate'], axis=1)
    df = pd.concat([df[df["label"]==0].sample(n=15100, random_state=42), df[df["label"]==1]]).reset_index(drop=True)
    st.dataframe(df.head(10))

    st.header("Tokenizing")
    st.write("Auf einem Beispiel zeige ich euch, wie Tokeniting funktioniert")
    example = df.iloc[1]["comment_text"]
    st.write("**Example:**\n")
    st.write(example, "  \n")
    
    tokens = word_tokenize(example, language="german")
    st.write("--------------Einzelne Tokens aus dem Example-------------------\n")
    st.write(tokens, format='plain', "  \n")
    
    tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
    st.write("--------------Tokens ohne Interpunktion-------------------\n")
    st.write(tokens_without_punctuation, format='plain', "\n")
    
    stop_words = stopwords.words("german")
    st.write("--------------Deutsche Stopwords der NLTK-Bibliothek-------------------\n")
    st.write(stop_words, format='plain', "\n")
    
    tokens_without_punctuation_and_stopwords = [i for i in tokens_without_punctuation if i not in stop_words]
    st.write("--------------Tokens ohne Interpunktion und Stopwords-------------------\n")
    st.write(tokens_without_punctuation_and_stopwords, format='plain')
    
    '''
    fig, ax = plt.subplots()
    for i, (key, value) in enumerate(all_mean_prices_for_brands_and_mil.items()):
        ax.plot(range(0, 300000, 1000), value, color=colors[i], marker='.', label=key)

    ax.set_title("Durchschnittspreise der Marken nach Kilometerzahl")
    ax.set_xlabel("Kilometerzahl")
    ax.set_ylabel("Durchschnittspreis")
    ax.legend()
    st.pyplot(fig)
    '''

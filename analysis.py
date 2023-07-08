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

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('omw') # Das deutsche Wörterbuch "GermaLemma"
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy
import subprocess
# Befehl ausführen, um das Modell herunterzuladen
subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"])

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
    st.write("**Einzelne Tokens:**")
    st.write(', '.join(tokens))
    
    tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
    st.write("**Tokens ohne Interpunktion:**")
    st.write(', '.join(tokens_without_punctuation))
    
    stop_words = stopwords.words("german")
    st.write("**Deutsche Stopwords der NLTK-Bibliothek:**")
    st.write(', '.join(stop_words))
    
    tokens_without_punctuation_and_stopwords = [i for i in tokens_without_punctuation if i not in stop_words]
    st.write("**Tokens ohne Interpunktion und Stopwords:**")
    st.write(', '.join(tokens_without_punctuation_and_stopwords))

    st.header("Stemming und Lemmatizing")
    st.write("Damit das Modell ähnliche Wörter wie 'Tisch' und 'Tische' als ein Wort erkennt und zählt, müssel alle Wörter im Text zu der Wurzelform gebracht werden.")
    st.write("Stemming-Methode löscht die Endungen von Wörtern, um ihre Grundform, auch Stamm genannt, zu erhalten.")

    def tokenized_and_stemmed(text):
        stemmer = SnowballStemmer(language="german")
        tokens = word_tokenize(text, language="german")
        tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
        tokens_without_punctuation_and_stopwords = [i for i in tokens_without_punctuation if i not in stop_words]
        stemmed_tokens = [stemmer.stem(i) for i in tokens_without_punctuation_and_stopwords]
        return stemmed_tokens
        
    st.write("**Stemmed Tokens:**")
    st.info(', '.join(tokenized_and_stemmed(example)))

    st.write("Weil mir das Ergebnis nicht sehr gefallen hat, habe ich weiter recherchiert und herausgefunden, dass Lemmatazing-Methode für die deutsche Sprache besser funktionieren sollte.")
    st.write("Lemmatizing-Methode sucht nach dem Wort in einem bestimmten Wörterbuch, in dem gleich steht, zu welcher Stammform dieses Wort gebracht werden soll.")

    def tokenized_and_lemmatized(text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text, language='german')
        tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
        tokens_without_punctuation_and_stopwords = [i for i in tokens_without_punctuation if i not in stop_words]
        lemmatized_tokes = [lemmatizer.lemmatize(i, 'v') for i in tokens_without_punctuation_and_stopwords]
        return lemmatized_tokes
    
    st.write("**Mit NLTK Lemmatized Tokens:**")
    st.info(', '.join(tokenized_and_lemmatized(example)))

    st.write("Außer NLTK-Bibliothek, habe ich Spacy-Bibliothek gefunden.")

    def tokenized_and_lemmatized_spacy(text):
        nlp = spacy.load('de_core_news_sm')
        tokens = nltk.word_tokenize(text, language='german')
        tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
        tokens_without_punctuation_and_stopwords = [i for i in tokens_without_punctuation if i not in stop_words]
        tokenized_text = " ".join(tokens_without_punctuation_and_stopwords)
        doc = nlp(tokenized_text)
        lemmas = [token.lemma_ for token in doc]
        return lemmas

    st.write("**Mit Spacy Lemmatized Tokens:**")
    st.info(', '.join(['möglicherweise', 'rechtfertigen', 'Umleitung', 'Element', 'existieren', 'fiktiv', 'Element', 'müssen', 'Leute', 'nachgeben', 'neu', 'Element', 'erfinden', 'krumm', 'befinden', '20:53', '24.', 'November', '2004', 'UTC']))

    st.write("Am Ende habe ich mich doch für Stemming entschieden. Aber es hat sich gelohnt, andere Methoden auszuprobieren.")

    st.header("Vectorizing")
    st.write("Ein Vectorizer verwandelt den Text in ein Vector aus Zeilen, das vom Modell verarbeitet werden kann.")
    st.write("Ich habe zwei Vectoriser aus der Sklearn-Bibliothek: ausprobiert und die Ergebnisse miteinander vergliechen.")
    st.write("Der CountVectorizer berücksichtigt nur die Häufigkeit der Wörter in einem Dokument und nicht ihre Bedeutung. Der TfidfVectorizer berücksichtigt sowohl die Häufigkeit der Wörter in einem Dokument als auch ihre Seltenheit im gesamten Korpus. Wörter, die in vielen Dokumenten vorkommen, erhalten ein niedrigeres Gewicht, während seltenere Wörter ein höheres Gewicht erhalten. Der TfidfVectorizer eignet sich besonders gut für Textklassifikationsaufgaben, da er es ermöglicht, Wörter zu identifizieren, die ein Dokument am besten charakterisieren und eine Unterscheidung zwischen relevanten und nicht relevanten Wörtern zu treffen.")

    st.write("**Classification Report für TfidfVectorizer:**")
    clas_report_tfidf = {'precision': {'0': 0.85,
                                       '1': 0.91,
                                       'accuracy': ""},
                         'recall': {'0': 0.92,
                                    '1': 0.83,
                                    'accuracy': ""},
                         'f1-score': {'0': 0.88,
                                      '1': 0.87,
                                      'accuracy': 0.88},
                         'support': {'0': 5036, 
                                     '1': 4926, 
                                     'accuracy': ""}}
    
    df_vec_tfidf = pd.DataFrame(clas_report_tfidf)
    st.dataframe(df_vec_tfidf)

    st.write("**Classification Report für CountVectorizer:**")
    clas_report_count = {'precision': {'0': 0.86,
                                       '1': 0.91,
                                       'accuracy': ""},
                         'recall': {'0': 0.92,
                                    '1': 0.85,
                                    'accuracy': ""},
                         'f1-score': {'0': 0.89,
                                      '1': 0.88,
                                      'accuracy': 0.88},
                         'support': {'0': 5036, 
                                     '1': 4926,
                                     'accuracy': ""}}
    
    df_vec_count = pd.DataFrame(clas_report_count)
    st.dataframe(df_vec_count)

    st.write("Für meine Aufgabe ist Recall für Klasse 1 (toxisch) entscheidend und CountVectorizer zeigt bessere Ergebnisse, deswegen habe ich weiter ihn benutzt.")

    st.header("Modelle")
    st.info("**LogisticRegression, NaiveBayes, RandomForest, GradienBoosting und SVM**")
    st.write("Zuerst muss ich die besten Parameter für jedes Modell finden.")
    st.write("Für **Logistic Regression** suche ich die besten Gewichte der Klassen")

    st.write("**Gewicht: {0: 1, 1: 1}:**")
    clas_report = {'precision': {'0': 0.86,
                                       '1': 0.91,
                                       'accuracy': ""},
                         'recall': {'0': 0.92,
                                    '1': 0.85,
                                    'accuracy': ""},
                         'f1-score': {'0': 0.89,
                                      '1': 0.88,
                                      'accuracy': 0.88},
                         'support': {'0': 5036, 
                                     '1': 4926,
                                     'accuracy': ""}}
    
    df_cr = pd.DataFrame(clas_report)
    st.dataframe(df_cr)

    st.write("**Gewicht: {0: 1, 1: 3}:**")
    clas_report = {'precision': {'0': 0.89,
                                 '1': 0.86,
                                 'accuracy': ""},
                   'recall': {'0': 0.85,
                              '1': 0.90,
                              'accuracy': ""},
                   'f1-score': {'0': 0.87,
                                '1': 0.88,
                                'accuracy': 0.87},
                   'support': {'0': 5036, 
                               '1': 4926,
                               'accuracy': ""}}
    
    df_cr = pd.DataFrame(clas_report)
    st.dataframe(df_cr)

    st.write("**Gewicht: {0: 1, 1: 5}:**")
    clas_report = {'precision': {'0': 0.91,
                                 '1': 0.83,
                                 'accuracy': ""},
                   'recall': {'0': 0.82,
                              '1': 0.91,
                              'accuracy': ""},
                   'f1-score': {'0': 0.86,
                                '1': 0.87,
                                'accuracy': 0.87},
                   'support': {'0': 5036, 
                               '1': 4926,
                               'accuracy': ""}}
    
    df_cr = pd.DataFrame(clas_report)
    st.dataframe(df_cr)

    st.write("Weil mir Recall für Klasse 1 (toxisch) wichtiger ist als Recall für Klasse 0, entscheide ich mich für die folgenden Gewichte: 0:1, 1:3.")

    st.write("Für **SupportVectorMachine** suche ich die besten 'C' und 'gamma' mit Hilfe von GridSearch und die Besten Parameter: **{'C': 10, 'gamma': 0.01}**")

    st.write("Für **Random Forest** suche ich mit Ellenbogenmethode die besten Anzahl und Tiefe der Bäume")
    mae_scores = [0.44, 0.47, 0.42, 0.32, 0.37, 0.36, 0.4, 0.41, 0.4, 0.38, 0.4, 0.41, 0.42, 0.4, 0.42, 0.43, 0.43, 0.43, 0.42, 0.41, 0.43, 0.44, 0.44, 0.44, 0.43, 0.44, 0.44, 0.44, 0.43, 0.44, 0.43, 0.44, 0.43]
    n_estimators_range = range(1, 100, 3)
    plt.plot(n_estimators_range, mae_scores)
    plt.xlabel('Zahl der Bäume')
    plt.ylabel('Anzahl der Fehler')
    plt.title('RandomForest - Optimale Zahl der Bäume')
    plt.show()

    
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

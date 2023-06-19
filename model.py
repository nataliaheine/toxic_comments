# IMPORTS:
# DATEN
import pandas as pd
from sklearn.model_selection import train_test_split

# TOKENIZING
import nltk

from nltk.tokenize import word_tokenize
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

import string

# STEMMING & LEMMATIZING
from nltk.stem import SnowballStemmer
nltk.download('omw-1.4')

# VECTORIZING
from sklearn.feature_extraction.text import CountVectorizer

# MODELLS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# SPEICHERN DES MODELLS
from joblib import dump
#--------------------------------------------------------------------------------------------------------------------

# DATEN LESEN UND VORBEREITEN
df = pd.read_csv("toxic_comments_de_unprocessed.csv")
df["label"] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].any(axis=1).astype(int)
df = df.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult','identity_hate'], axis=1)
df = df[df["label"]==0].sample(n=15100, random_state=42).append(df[df["label"]==1]).reset_index(drop=True)

# TOKENIZING & STEMMING FUNKTION
stop_words = stopwords.words("german")
def tokenized_and_stemmed(text):
    stemmer = SnowballStemmer(language="german")
    tokens = word_tokenize(text, language="german")
    tokens_without_punctuation = [i for i in tokens if i not in string.punctuation]
    tokens_without_punctuation_and_stopwords = [i for i in tokens_without_punctuation if i not in stop_words]
    stemmed_tokens = [stemmer.stem(i) for i in tokens_without_punctuation_and_stopwords]
    return stemmed_tokens

# MODEL ERSTELLEN UND TRINIEREN
model_pipeline = Pipeline([
    ("vectorizer", CountVectorizer(tokenizer=lambda x: tokenized_and_stemmed(x), lowercase=False)),
    ("model", LogisticRegression(class_weight={0: 1, 1: 3}))
])
model_pipeline.fit(df["comment_text"], df["label"])
dump(model_pipeline, 'model_toxic_comments.joblib')
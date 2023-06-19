import streamlit as st
from joblib import load

model_pipeline = load('model_toxic_comments.joblib') 

# PREDICT COMMENT FUNKTION
def predict_comment(text):
    klasse = model_pipeline.predict([text])
    proba = model_pipeline.predict_proba([text])
    if klasse == 1:
        return klasse, round(proba[0][1]*100, 2)
    else:
        return klasse, round(proba[0][0]*100, 2)
    
st.title("Toxic comments")
st.write(f"Du kannst einen Kommentar nach Toxizität überprüfen")
text = st.text_input("Schreibe hier den Kommentartext")

if text != "":
    klasse, proba = model.predict_comment(text)
    st.write(f"Der Kommentar '{text}' ist **{'toxisch' if klasse==1 else 'nicht toxisch'}** mit der Wahrscheinlichkeit von **{proba}%**")
import model
import streamlit as st

st.title("Toxic comments")
st.write(f"Du kannst einen Kommentar nach Toxizität überprüfen")
text = st.text_input("Schreibe hier den Kommentartext")

if text != "":
    klasse, proba = model.predict_comment(text)
    st.write(f"Der Kommentar '{text}' ist **{'toxisch' if klasse==1 else 'nicht toxisch'}** mit der Wahrscheinlichkeit von **{proba}%**")
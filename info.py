import streamlit as st

def start():
    st.title("Toxic Comments Classification")

    st.success("Für dieses Projekt habe ich ein Dataset mit gelabelten Kommentaren auf Deutsch gefunden.  \n\nDas Dataset findet man unter folgendem [Link](https://www.kaggle.com/datasets/shubheshswain/jigsaw-toxic-comment-classification-german).")
            
    st.success("Unter diesem [Link](https://github.com/nataliaheine/toxic_comments) kann man das gesamte Projekt auf GitHub anschauen.  \n\nDort findet man unter anderem ein Jupyter Notebook, in dem zu sehen ist, wie das Modell ausgewählt, trainiert und gespeichert wurde. Teile dieses Notebook findet man auf der Seite 'Datenanalyse & Modellauswahl'")

    st.success("Wichtige Info, falls du gleich das Modell ausprobieren willst:  \n\nWenn mein Modell auf einer echten Webseite benutzt würde, wären die als toxisch erkannte Kommentare einem Mitarbeiter geschickt, der entscheiden sollte, was damit weiter gemacht werden soll.  \n\nDaher ist es nicht schlimm, wenn 'gute' Kommentare als 'schlechte' erkannt werden, aber andersum wäre es nicht erwünscht. Aufgrundessen wurde das Modell so trainiert, dass sie oft harmlose Kommentare als toxisch erkennt, aber so gut wie nie toxische Kommentare als nicht toxisch markiert. Also wundere dich nicht, wenn 'Wie wird das Wetter morgen sein?' mit 55% Wahrscheinlichekit als toxisch erkannt wird. Liegt an den Einstellungen beim Training UND an dem Dataset selbst, denn dort wurden bereits minimal unangenehme Kommentare auch als toxisch gelabelt.")

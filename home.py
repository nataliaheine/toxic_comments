from matplotlib import widgets
import streamlit as st

import model
import info
import analysis

from model import tokenized_and_stemmed

pages = {
    "1. Info": info,
    "2. Datenanalyse & Modellauswahl": analysis,
    "3. Modell": model
}

st.sidebar.title("Seitenmenü")
select = st.sidebar.radio("", list(pages.keys()))
pages[select].start()

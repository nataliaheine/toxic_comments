import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("toxic_comments_de_unprocessed.csv")

def start():
    st.title("Datenanalyse")


    st.info("In diesem Projekt wurden die Daten von der Webseite 'AutoScout24' analysiert.  \n\nDas Dataset findet man unter folgendem [Link](https://www.kaggle.com/datasets/ander289386/cars-germany).")


    st.header("Wie viele Autos wurden verkauft?  Über welchen Zeitraum?  Welche Marken sind erfasst?")
    
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
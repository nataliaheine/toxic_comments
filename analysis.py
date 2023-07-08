import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("toxic_comments_de_unprocessed.csv")

def start():
    st.title("Datenanalyse & Modellauswahl")


    st.info("Dataset findet man unter folgendem [Link](https://www.kaggle.com/datasets/shubheshswain/jigsaw-toxic-comment-classification-german).")


    st.header("So sehen die ursprunglichen Daten aus:")
    st.dataframe(df.head(10))
    
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

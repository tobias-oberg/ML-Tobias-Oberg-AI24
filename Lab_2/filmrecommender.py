import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests


#my_api_key = "f43b87de9064ce5282fc85bc77c5d2cc"



@st.cache_data # cachar data så att det inte laddas om varje gång
def load_data():
    movies = pd.read_csv("movies.csv")
    tags = pd.read_csv("tags.csv")
    
    return movies, tags

# Rensar specialtecken och behåller endast bokstäver och siffror
# https://www.geeksforgeeks.org/python-remove-all-characters-except-letters-and-numbers/
def clean_data(text):
        return re.sub("[^a-zA-Z0-9]"," ", text).strip()


def clean_combine_data(movies, tags):
    # Gruppera och slå ihop tags
    tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.drop_duplicates().astype(str))).reset_index()
    movies = movies.merge(tags_grouped, on="movieId", how="left")
   
    # Fyll i NaN-värden med tomma strängar
    movies["tag"] = movies["tag"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    
    # Rensa texten i title, tag och genres, använder clean_data - funktionen 
    movies["title"] = movies["title"].apply(clean_data)
    movies["tag"] = movies["tag"].apply(clean_data)
    
    
    movies["genres"] = movies["genres"].apply(clean_data)
    
    # Kombinera genres och tags 
    movies["combined_genres_tag"] = movies["genres"] + " " + movies["tag"]

    return movies




# Skapa en TF-IDF-matris med n-gram och stopwords
@st.cache_resource # cachar resursen så att den inte laddas om varje gång
def vectorizer_matrix(movies):
   vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
   tfidf = vectorizer.fit_transform(movies["combined_genres_tag"]) 
   return vectorizer, tfidf


     
# Beräkna likheten mellan en given film och alla andra filmer i datasetet
# Använd cosine similarity för att beräkna likheten mellan TF-IDF-vektorerna
# Stort dataset så istället jämförs filmerna 1 mot 1
def similar_movies(tfidf, vectorizer, query, movies):
    query = clean_data(query) 
    query_vec = vectorizer.transform([query]) 

    sim_score = cosine_similarity(query_vec, tfidf).flatten() # flatten gör att den blir en array istället för en matris
    indices = np.argsort(sim_score)[-5:][::-1] # sorterar i descending order och tar de 5 mest lika filmerna
    results = movies.iloc[indices] # hämtar de 5 mest lika filmerna från movies dataframe

    return results[["title"]] # returnerar title för de 5 mest lika filmerna


st.markdown(" # Filmrekommendationer")
st.write("Sök efter en film för att få rekommendationer")


movies, tags = load_data() 
movies = clean_combine_data(movies,tags) 
vectorizer, tfidf = vectorizer_matrix(movies) 

movie_name = st.text_input("Skriv in en film: :movie_camera:", "")




if movie_name:
        results = similar_movies(tfidf, vectorizer, movie_name, movies)
        results = results.rename(columns={"title": "Titlar:"})
        st.write("Här är 5 liknande filmer:")
        st.dataframe(results, hide_index=True)
        
       



                       


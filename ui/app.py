import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on similarity")


# Load data and models
@st.cache_resource
def load_assets():
    movies = pd.read_csv("data/movies_clean.csv")

    with open("models/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("models/indices.pkl", "rb") as f:
        indices = pickle.load(f)

    return movies, tfidf_matrix, indices


movies, tfidf_matrix, indices = load_assets()


# Content-based recommendation
def recommend_movies(title, top_n=10):

    if title not in indices:
        return []

    idx = indices[title]

    scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = np.argsort(scores)[::-1][1:top_n+1]

    return movies.iloc[top_indices]["title"].tolist()


# UI Inputs
movie_title = st.text_input("Movie Title", "Toy Story (1995)")
top_n = st.slider("Number of recommendations", 5, 20, 10)


# Recommendation button
if st.button("Recommend"):

    results = recommend_movies(movie_title, top_n)

    if results:
        st.subheader("Recommended Movies")
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("Movie not found in dataset")

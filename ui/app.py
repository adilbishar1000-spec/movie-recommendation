import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests

TMDB_API_KEY = "022e719fcd53040c1f71f2d7314b2cfe"
def fetch_poster(movie_title):

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_title
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["results"]:
        poster_path = data["results"][0]["poster_path"]
        return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return None
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")

st.write("Get movie recommendations based on similarity using a content-based recommendation system.")


# ---- About the dataset section ----
with st.expander("ℹ️ About this recommendation system"):
    st.write("""
    This movie recommendation system is built using the **MovieLens dataset**.

    **Dataset details:**
    - 🎞 Movies included: ~87,000+
    - 📅 Year coverage: approximately **1900 – 2019**
    - 🌍 Primary language: **English**
    - 🧠 Recommendation method: **Content-based filtering using TF-IDF similarity**
    
    The system analyzes movie titles and metadata to find movies that are similar to the one you enter.
    """)


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
st.subheader("🔎 Find Similar Movies")

movie_title = st.text_input("Movie Title", "Toy Story (1995)")
top_n = st.slider("Number of recommendations", 5, 20, 10)


# Recommendation button
if st.button("Recommend"):

    results = recommend_movies(movie_title, top_n)

    if results:
        st.subheader("🎥 Recommended Movies")
       for movie in results:

    poster = fetch_poster(movie)

    col1, col2 = st.columns([1,3])

    with col1:
        if poster:
            st.image(poster)

    with col2:
        st.subheader(movie)
    else:
        st.warning("Movie not found in dataset. Try another title.")

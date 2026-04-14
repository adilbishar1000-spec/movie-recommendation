import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on similarity.")


# -----------------------
# TMDB API KEY
# -----------------------
TMDB_API_KEY = "022e719fcd53040c1f71f2d7314b2cfe"


# -----------------------
# Dataset Information
# -----------------------
with st.expander("ℹ️ About this system"):
    st.write("""
    This recommendation system is built using the MovieLens dataset.

    **Dataset Details**
    - 🎞 Movies: ~87,000+
    -  Year coverage: 1900 – 2019
    -  Movie Language: English
    """)


# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_assets():

    movies = pd.read_csv("data/movies_clean.csv")

    with open("models/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open("models/indices.pkl", "rb") as f:
        indices = pickle.load(f)

    return movies, tfidf_matrix, indices


movies, tfidf_matrix, indices = load_assets()


# -----------------------
# Recommendation Function
# -----------------------
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


# -----------------------
# Poster Fetch Function
# -----------------------
def fetch_poster(movie_title):

    clean_title = re.sub(r"\(\d{4}\)", "", movie_title).strip()

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data["results"]:
            poster_path = data["results"][0]["poster_path"]

            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except:
        pass

    return None


# -----------------------
# User Inputs
# -----------------------
st.subheader("🔎 Find Similar Movies")

movie_title = st.text_input("Movie Title", "Toy Story (1995)")
top_n = st.slider("Number of recommendations", 5, 20, 10)


# -----------------------
# Recommendation Button
# -----------------------
if st.button("Recommend"):

    results = recommend_movies(movie_title, top_n)

    if results:

        st.subheader("🎥 Recommended Movies")

        cols = st.columns(5)

        for i, movie in enumerate(results):

            poster = fetch_poster(movie)

            with cols[i % 5]:

                if poster:
                    st.image(poster, use_container_width=True)

                st.caption(movie)

    else:
        st.warning("Movie not found in dataset.")

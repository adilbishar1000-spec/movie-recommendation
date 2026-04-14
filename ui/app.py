import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations")

user_id = st.number_input("User ID", min_value=1, step=1, value=1)
movie_title = st.text_input("Movie Title", "Toy Story (1995)")
top_n = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    params = {
        "user_id": user_id,
        "movie_title": movie_title,
        "top_n": top_n
    }

    try:
        response = requests.get(API_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            st.subheader("Recommended Movies")
            for i, movie in enumerate(data["recommendations"], start=1):
                st.write(f"{i}. {movie}")
        else:
            st.error("API returned an error")

    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

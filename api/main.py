from fastapi import FastAPI
import pickle

from recommender.content_based import ContentRecommender
from recommender.collaborative import CollaborativeRecommender
from recommender.hybrid import HybridRecommender

app = FastAPI(title="Movie Recommendation API")

# Load shared data
with open("models/user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

with open("models/user_id_map.pkl", "rb") as f:
    user_id_map = pickle.load(f)

content_rec = ContentRecommender(
    "data/movies_clean.csv",
    "models/tfidf_vectorizer.pkl",
    "models/tfidf_matrix.pkl",
    "models/indices.pkl"
)

collab_rec = CollaborativeRecommender(
    "data/ratings_clean.csv",
    "models/knn_model.pkl",
    user_item_matrix,
    user_id_map
)

hybrid_rec = HybridRecommender(content_rec, collab_rec)


@app.get("/")
def root():
    return {"status": "api running"}


@app.get("/recommend")
def recommend(user_id: int, movie_title: str, top_n: int = 10):
    recs = hybrid_rec.recommend(
        user_id=user_id,
        movie_title=movie_title,
        top_n=top_n
    )

    return {
        "user_id": user_id,
        "movie_title": movie_title,
        "recommendations": recs
    }

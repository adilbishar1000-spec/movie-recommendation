import pickle
import pandas as pd
import numpy as np


class CollaborativeRecommender:
    def __init__(self, ratings_path, knn_path, user_item_matrix, user_id_map):
        self.ratings = pd.read_csv(ratings_path)

        with open(knn_path, "rb") as f:
            self.knn = pickle.load(f)

        self.user_item_matrix = user_item_matrix
        self.user_id_map = user_id_map

    def recommend(self, user_id, top_n=10):
        if user_id not in self.ratings["userId"].values:
            return []

        user_idx = (
            self.ratings["userId"]
            .astype("category")
            .cat.categories.get_loc(user_id)
        )

        _, indices = self.knn.kneighbors(
            self.user_item_matrix[user_idx], n_neighbors=6
        )

        similar_users = indices.flatten()[1:]
        movie_scores = {}

        for u in similar_users:
            sim_user_id = self.user_id_map[u]
            sim_ratings = self.ratings[self.ratings["userId"] == sim_user_id]

            for _, row in sim_ratings.iterrows():
                movie_scores[row["movieId"]] = movie_scores.get(
                    row["movieId"], 0
                ) + row["rating"]

        return sorted(movie_scores, key=movie_scores.get, reverse=True)[:top_n]

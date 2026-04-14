import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, movies_path, tfidf_path, matrix_path, indices_path):
        self.movies = pd.read_csv(movies_path)

        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)

        with open(matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)

        with open(indices_path, "rb") as f:
            self.indices = pickle.load(f)

    def recommend(self, title, top_n=10):
        if title not in self.indices:
            return []

        idx = self.indices[title]
        movie_vector = self.tfidf_matrix[idx]

        scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][1:top_n+1]

        return self.movies.iloc[top_indices]["title"].tolist()

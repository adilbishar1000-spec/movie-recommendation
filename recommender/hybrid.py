class HybridRecommender:
    def __init__(self, content_rec, collab_rec):
        self.content_rec = content_rec
        self.collab_rec = collab_rec

    def recommend(self, user_id, movie_title, top_n=10):
        content_movies = self.content_rec.recommend(movie_title, top_n=30)
        collab_ids = set(self.collab_rec.recommend(user_id, top_n=30))

        boosted = []
        for title in content_movies:
            score = 1.0
            if title in collab_ids:
                score += 1.0
            boosted.append((title, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in boosted[:top_n]]

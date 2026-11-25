from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


class MovieRecommender:
    def __init__(self, project_root: Path | None = None):
        self.root = project_root or Path(__file__).resolve().parents[2]
        self.model_dir = self.root / "models"
        self.movies_path = self.root / "data" / "raw" / "ml-20m" / "movies.csv"
        self.train_path = self.root / "data" / "processed" / "splits" / "train.parquet"

        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Movie metadata & genres
        self.movies = pd.read_csv(self.movies_path)
        self.movies["genres_clean"] = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
            .replace("(no genres listed)", "")
        )
        self.content_movie_to_idx = pd.Series(
            self.movies.index, index=self.movies["movieId"]
        ).to_dict()

        # 2. Popularity baseline
        self.popularity_df = pd.read_parquet(
            self.model_dir / "weighted_popularity_baseline.parquet"
        )

        # 3. SVD artifacts
        self.user_factors = np.load(self.model_dir / "svd_user_factors.npy")
        self.item_factors = np.load(self.model_dir / "svd_item_factors.npy")
        mappings = joblib.load(self.model_dir / "svd_mappings.joblib")
        self.user_to_idx = mappings["user_to_idx"]
        self.movie_to_idx = mappings["movie_to_idx"]
        self.idx_to_movie = mappings["idx_to_movie"]

        # 4. Content-based TF-IDF
        self.tfidf = joblib.load(self.model_dir / "tfidf_vectorizer.joblib")
        self.tfidf_matrix = self.tfidf.transform(self.movies["genres_clean"])

    def _get_watched_movies(self, user_id: int) -> set:
        if not self.train_path.exists():
            return set()
        con = duckdb.connect()
        query = f"""
        SELECT movieId
        FROM read_parquet('{self.train_path.as_posix()}')
        WHERE userId = {int(user_id)}
        """
        watched = con.execute(query).df()["movieId"].tolist()
        con.close()
        return set(watched)

    def recommend_popular(self, n: int = 10) -> pd.DataFrame:
        return self.popularity_df.head(n)[["movieId", "title", "genres", "weighted_score"]]

    def recommend_by_genres(self, selected_genres: list[str], n: int = 10) -> pd.DataFrame:
        if not selected_genres:
            return self.recommend_popular(n)

        genre_query = " ".join(selected_genres)
        query_vec = self.tfidf.transform([genre_query])
        sim_scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argpartition(sim_scores, -n)[-n:]
        top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]

        result = self.movies.iloc[top_indices][["movieId", "title", "genres"]].copy()
        result["match_score"] = np.round(sim_scores[top_indices], 4)
        return result

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_to_idx:
            return self.recommend_popular(n)

        u_idx = self.user_to_idx[user_id]
        scores = self.user_factors[u_idx] @ self.item_factors.T

        watched_ids = self._get_watched_movies(user_id)
        for m_id in watched_ids:
            if m_id in self.movie_to_idx:
                scores[self.movie_to_idx[m_id]] = -np.inf

        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        rec_df = pd.DataFrame({
            "movieId": [self.idx_to_movie[i] for i in top_indices],
            "predicted_score": np.round(scores[top_indices], 3),
        })
        return rec_df.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")

    def recommend_similar_hybrid(
        self, movie_id: int, n: int = 10, svd_weight: float = 0.7
    ) -> pd.DataFrame:
        if movie_id not in self.movie_to_idx or movie_id not in self.content_movie_to_idx:
            return pd.DataFrame()

        # 1. SVD item similarity
        svd_idx = self.movie_to_idx[movie_id]
        svd_sim = cosine_similarity(
            self.item_factors[svd_idx].reshape(1, -1), self.item_factors
        ).flatten()

        # 2. Content (genre) similarity aligned to SVD movie order
        content_idx = self.content_movie_to_idx[movie_id]
        raw_content_sim = linear_kernel(
            self.tfidf_matrix[content_idx], self.tfidf_matrix
        ).flatten()

        content_sim = np.zeros(len(self.movie_to_idx), dtype=np.float32)
        for m_id, s_idx in self.movie_to_idx.items():
            c_idx = self.content_movie_to_idx.get(m_id)
            if c_idx is not None:
                content_sim[s_idx] = raw_content_sim[c_idx]

        # 3. Combine both scores
        hybrid_scores = (svd_weight * svd_sim) + ((1 - svd_weight) * content_sim)
        hybrid_scores[svd_idx] = -np.inf

        top_indices = np.argpartition(hybrid_scores, -n)[-n:]
        top_indices = top_indices[np.argsort(hybrid_scores[top_indices])[::-1]]

        rec_df = pd.DataFrame({
            "movieId": [self.idx_to_movie[i] for i in top_indices],
            "hybrid_score": np.round(hybrid_scores[top_indices], 4),
        })
        return rec_df.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")

    def recommend_for_new_user(
            self,
            user_ratings: dict[int, float] | None = None,
            selected_genres: list[str] | None = None,
            n: int = 10,
        ) -> tuple[pd.DataFrame, str]:
            user_ratings = user_ratings or {}
            selected_genres = selected_genres or []

            # Case 1: Completely new user (no ratings, no genres) -> Popularity
            if not user_ratings and not selected_genres:
                return self.recommend_popular(n), "Popularity Baseline (No user input yet)"

            # Case 2: Only genres selected -> Content-Based Filtering
            if not user_ratings and selected_genres:
                return (
                    self.recommend_by_genres(selected_genres, n),
                    "Content-Based Filtering (Matched using selected genres)",
                )

            # Case 3: User rated at least 1 movie -> Real-Time SVD Folding-In
            new_user_vec = np.zeros(self.item_factors.shape[1], dtype=np.float32)
            rated_svd_indices = []

            for m_id, rating in user_ratings.items():
                if m_id in self.movie_to_idx:
                    idx = self.movie_to_idx[m_id]
                    new_user_vec += float(rating) * self.item_factors[idx]
                    rated_svd_indices.append(idx)

            if not rated_svd_indices:
                return self.recommend_by_genres(selected_genres, n), "Content-Based Fallback"

            svd_scores = new_user_vec @ self.item_factors.T

            # Normalize SVD scores to 0-1 range
            s_min, s_max = svd_scores.min(), svd_scores.max()
            if s_max > s_min:
                svd_scores = (svd_scores - s_min) / (s_max - s_min)

            strategy = "Real-Time SVD Collaborative Filtering (Based on your ratings)"

            # If genres are also selected, blend 75% SVD + 25% Genre match (Hybrid)
            if selected_genres:
                genre_query = " ".join(selected_genres)
                query_vec = self.tfidf.transform([genre_query])
                raw_content = linear_kernel(query_vec, self.tfidf_matrix).flatten()

                content_scores = np.zeros(len(self.movie_to_idx), dtype=np.float32)
                for m_id, s_idx in self.movie_to_idx.items():
                    c_idx = self.content_movie_to_idx.get(m_id)
                    if c_idx is not None:
                        content_scores[s_idx] = raw_content[c_idx]

                final_scores = (0.75 * svd_scores) + (0.25 * content_scores)
                strategy = "Real-Time Hybrid (75% SVD Ratings + 25% Selected Genres)"
            else:
                final_scores = svd_scores

            # Hide movies the new user just rated
            for idx in rated_svd_indices:
                final_scores[idx] = -np.inf

            top_indices = np.argpartition(final_scores, -n)[-n:]
            top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

            rec_df = pd.DataFrame({
                "movieId": [self.idx_to_movie[i] for i in top_indices],
                "match_score": np.round(final_scores[top_indices], 4),
            })
            rec_df = rec_df.merge(
                self.movies[["movieId", "title", "genres"]], on="movieId", how="left"
            )
            return rec_df, strategy
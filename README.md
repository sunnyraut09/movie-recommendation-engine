# Personalized Movie Recommendation System

**Live App:** [Click here to open the Movie Recommender](https://movie-recommendation-engine-j6yp5qec3rq9rtd72dwaiv.streamlit.app/)

## 1. Problem Statement & Real-World Relevance
With thousands of movies available on streaming platforms, users suffer from choice overload and struggle to find movies they will enjoy. This project builds an end-to-end **Personalized Movie Recommendation System** designed for both **new users** (cold-start) and **returning users**. It reduces search time and increases user engagement by suggesting relevant movies based on viewing patterns and genres.

## 2. Dataset Documentation
* **Source:** [MovieLens 20M Dataset (GroupLens)](https://grouplens.org/datasets/movielens/20m/)
* **Size:** 20,000,263 ratings across 138,493 users and 27,278 movies (26,744 rated movies).
* **Features:** `userId`, `movieId`, `rating` (0.5 to 5.0), `timestamp`, `title`, and pipe-separated `genres`.
* **Preprocessing & Memory Optimization (8 GB RAM):**
  * Processed the 508.73 MB CSV in 500,000-row chunks and downcasted types (`int32`, `float32`), compressing storage to 138.67 MB using Parquet.
  * Used SciPy CSR Sparse Matrices (151.64 MB instead of ~14.8 GB dense matrix).
* **Data Leakage Prevention:** Used **Leave-One-Out Temporal Splitting** via DuckDB (`ROW_NUMBER() OVER (PARTITION BY userId ORDER BY timestamp DESC)`). Each user's latest rating goes to `test.parquet` and earlier ratings go to `train.parquet` (0 overlapping interactions).
* **Data Limitations:** High matrix sparsity (99.46% empty), strong popularity imbalance (long-tail distribution), and positive-rating bias (most ratings are 3.0 to 4.0).

## 3. Algorithms Implemented
1. **Simple Popularity Baseline:** Ranks movies by total rating count in the training set.
2. **Bayesian Weighted Popularity:** Uses the IMDb weighted rating formula ($m = 1000$) to balance rating count and average rating.
3. **Item-Based Collaborative Filtering (KNN):** Uses Cosine Similarity on the sparse item-user matrix to find movies rated similarly by users.
4. **Matrix Factorization (TruncatedSVD):** Compresses the user-item matrix into 50 latent factors for fast, personalized scoring.
5. **Content-Based Filtering (TF-IDF):** Converts movie genres into TF-IDF vectors to match user genre preferences.
6. **Real-Time Hybrid Engine:** Combines SVD Folding-In (75% weight) with TF-IDF genre similarity (25% weight) to give instant recommendations to brand-new users.

## 4. Model Comparison, Evaluation & Error Analysis
Evaluated using **Recall@10** on relevant test interactions ($\text{rating} \ge 4.0$):

| Model | Relevant Test Ratings | Hits | Recall@10 |
| :--- | :---: | :---: | :---: |
| Simple Popularity (Count) | 80,321 | 3,266 | 4.07% |
| Weighted Popularity (Bayesian) | 80,321 | 2,252 | 2.80% |
| **Matrix Factorization (SVD)** | **5,000 (sampled)** | **681** | **13.62%** |

**Error Analysis & Practical Insights:**
* **Why Simple Popularity beat Bayesian Popularity on Recall@10:** Simple Popularity recommends blockbuster movies watched by the most people, whereas Bayesian Popularity boosts older, critically acclaimed classics watched by fewer users.
* **Why SVD won by 3x:** SVD captures hidden user tastes rather than recommending the same global list to everyone.
* **Cold-Start Limitation:** Pure SVD cannot recommend to users with zero rating history. We solved this in the prototype using a 3-tier fallback: Popularity (no input) -> Content-Based (genres only) -> Real-Time SVD/Hybrid (live ratings).

## 5. Ethics, Privacy & Responsible Use
* **Popularity Bias & Filter Bubbles:** Collaborative filtering tends to over-recommend already-popular movies and trap users in narrow genres. Combining content-based features and weighted scoring helps surface diverse titles.
* **Privacy:** MovieLens uses anonymized numeric user IDs with no personally identifiable information (PII). In a production system, user watch history must be encrypted and users should be able to delete their data.
* **Responsible Use:** Recommendation systems should avoid manipulative engagement loops and clearly explain why an item is recommended (e.g., showing the active strategy in our UI).

## 6. Reproducibility & How to Run Locally
```bash
# 1. Clone the repository
git clone YOUR_GITHUB_REPO_URL
cd movie-recommendation-engine

# 2. Install dependencies using uv
uv sync

# 3. Run the Streamlit web application
uv run streamlit run app.py
import streamlit as st
from src.recommendation.recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")


@st.cache_resource
def load_recommender():
    return MovieRecommender()


recommender = load_recommender()

st.title("Personalized Movie Recommendation System")
st.write("Discover movies tailored to your taste using Hybrid Machine Learning (SVD + Content-Based + Popularity).")

top_n = st.slider("Number of recommendations to display", min_value=5, max_value=20, value=10)

tab_new, tab_similar = st.tabs([
    "Get Recommendations",
    "Find Similar Movies",
])

# --- TAB 1: NEW USER MAIN SCREEN ---
with tab_new:
    st.subheader("Find Your Next Favorite Movie")

    with st.container(border=True):
        st.markdown("**How to use this recommender (3 simple ways):**")
        st.markdown(
            """
            1. **Best results (Hybrid):** Pick 2 to 5 movies you have already watched, rate them, and select your favorite genres.
            2. **Quick genre match:** Skip the movies and just pick your favorite genres.
            3. **Not sure what to pick?** Leave everything empty and click **Get My Recommendations** to see the highest-rated popular movies.
            """
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Step 1: Select movies you have watched (Optional)**")
        movie_titles = recommender.movies["title"].tolist()
        picked_movies = st.multiselect(
            "Search and pick 1 to 5 movies:",
            options=movie_titles,
            max_selections=5,
            placeholder="e.g., Inception, Matrix, Toy Story...",
        )

        user_ratings = {}
        if picked_movies:
            st.markdown("**Rate your selected movies:**")
            for title in picked_movies:
                m_id = int(
                    recommender.movies.loc[
                        recommender.movies["title"] == title, "movieId"
                    ].values[0]
                )
                user_ratings[m_id] = st.slider(
                    f"Your rating for {title}",
                    min_value=0.5,
                    max_value=5.0,
                    value=4.5,
                    step=0.5,
                )

    with col2:
        st.markdown("**Step 2: Pick favorite genres (Optional)**")
        all_genres = [
            "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Horror",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ]
        selected_genres = st.multiselect(
            "Choose genres you enjoy:",
            options=all_genres,
        )

    if st.button("Get My Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            results, strategy_used = recommender.recommend_for_new_user(
                user_ratings=user_ratings,
                selected_genres=selected_genres,
                n=top_n,
            )
        st.info(f"**Active Strategy:** {strategy_used}")
        st.dataframe(results, use_container_width=True, hide_index=True)

# --- TAB 2: SIMILAR MOVIES (HYBRID) ---
with tab_similar:
    st.subheader("Find Similar Movies (SVD + Genres)")
    st.write("Pick any movie you like to find similar movies based on viewer patterns and genres.")

    movie_titles = recommender.movies["title"].tolist()
    selected_title = st.selectbox("Select or type a movie name:", movie_titles, index=0)

    svd_weight = st.slider(
        "Balance: Viewer Taste (SVD) vs. Genre Match",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
    )

    if st.button("Find Similar Movies", type="primary"):
        movie_id = int(
            recommender.movies.loc[
                recommender.movies["title"] == selected_title, "movieId"
            ].values[0]
        )
        with st.spinner("Finding similar movies..."):
            results = recommender.recommend_similar_hybrid(
                movie_id, n=top_n, svd_weight=svd_weight
            )

        if results.empty:
            st.warning("This movie does not have enough ratings in the training data.")
        else:
            st.dataframe(results, use_container_width=True, hide_index=True)

# --- HIDDEN EVALUATOR DEMO (COLLAPSED BY DEFAULT) ---
st.divider()
with st.expander("Evaluator Demo: Test Model on Historical MovieLens Users"):
    st.caption(
        "This section is for project evaluation only. It tests the trained SVD model on the 138,493 historical users from the MovieLens 20M dataset."
    )
    user_id = st.number_input("Enter Historical User ID (1 to 138493)", min_value=1, max_value=138493, value=1)

    if st.button("Run Historical User Demo"):
        with st.spinner("Generating recommendations..."):
            results = recommender.recommend_for_user(int(user_id), n=top_n)
        st.dataframe(results, use_container_width=True, hide_index=True)
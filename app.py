import streamlit as st
from src.recommendation.recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --stone: #f4f1ea;
        --paper: #fbfaf7;
        --ink: #242321;
        --muted: #777269;
        --line: #ddd8ce;
        --accent: #a84b3f;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--stone);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 3.25rem;
        padding-bottom: 5rem;
    }

    [data-testid="stMain"] {
        color: var(--ink);
    }

    h1, h2, h3,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: -0.035em;
        font-weight: 700;
    }

    h1 {
        max-width: 820px;
        font-size: 2.35rem !important;
        line-height: 1.08;
        margin-bottom: 0.65rem !important;
    }

    [data-testid="stTitle"] {
        border-bottom: 1px solid var(--line);
        padding-bottom: 1.5rem;
        margin-bottom: 1.25rem;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--stone) !important;
        color: var(--ink) !important;
    }

    [data-testid="stHeader"] {
        background: var(--stone) !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"] p,
    label,
    [data-testid="stWidgetLabel"] p {
        color: var(--ink);
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] p {
        line-height: 1.55;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted);
        font-size: 0.82rem;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stCaptionContainer"],
    [data-testid="stText"],
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricLabel"] {
        color: var(--ink) !important;
    }

    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {
        font-size: 1.35rem !important;
        line-height: 1.2;
    }

    [data-testid="stTabs"] {
        margin-top: 0.25rem;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 2.25rem;
        border-bottom: 1px solid var(--line);
    }

    [data-testid="stTabs"] [role="tab"] {
        border: none;
        border-radius: 0;
        background: transparent;
        color: var(--muted);
        padding: 0.9rem 0 0.75rem;
        font-weight: 700;
        font-size: 0.95rem;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--ink);
        border-bottom: 2px solid var(--accent);
    }

    [data-testid="stTabs"] [role="tab"] p,
    [data-testid="stTabs"] [role="tab"] span {
        color: inherit !important;
    }

    [data-testid="stTab"] {
        padding-top: 2.25rem;
    }

    [data-testid="stColumn"] {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    [data-testid="stColumn"]:first-child {
        padding-left: 0;
    }

    [data-testid="stColumn"]:last-child {
        padding-right: 0;
    }

    [data-testid="stContainer"] {
        background: transparent;
    }

    [data-testid="stContainer"][border="true"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
        background: rgba(251, 250, 247, 0.48);
    }

    [data-testid="stSlider"] {
        border-bottom: 1px solid var(--line);
        padding: 0.15rem 0 1.15rem;
        margin-bottom: 0.35rem;
    }

    [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
        color: #4d4a44;
        font-size: 0.84rem;
        font-weight: 650;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--accent) !important;
        border: 2px solid var(--paper) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] div {
        background: var(--line);
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: #ede1dc;
        border: 1px solid #bd8177;
        color: #71382f;
        border-radius: 999px;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
        color: #71382f;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: var(--paper) !important;
        border-color: var(--line) !important;
        border-radius: 2px !important;
        color: var(--ink) !important;
    }

    [data-baseweb="select"] input,
    [data-baseweb="select"] span,
    [data-baseweb="input"] input,
    [data-baseweb="tag"] span,
    [data-testid="stNumberInput"] input {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 3.25rem;
        border: 1px solid var(--ink);
        border-radius: 2px;
        background: var(--ink);
        color: #fffdf7;
        opacity: 1;
        font-weight: 750;
        letter-spacing: 0.01em;
        transition: background 160ms ease, border-color 160ms ease;
    }

    .stButton > button:hover,
    .stButton > button:focus {
        border-color: #151514;
        background: #151514;
        color: #fffdf7;
    }

    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #fffdf7 !important;
        -webkit-text-fill-color: #fffdf7 !important;
    }

    [data-testid="stBaseButton-secondary"] {
        width: 100%;
    }

    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"] {
        background: transparent;
        border: none;
    }

    [data-testid="stDataFrame"] div[role="table"],
    [data-testid="stDataFrameResizable"] div[role="table"] {
        border: none;
    }

    [data-testid="stDataFrame"] [role="row"],
    [data-testid="stDataFrameResizable"] [role="row"] {
        border: none;
    }

    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrameResizable"] [role="columnheader"] {
        background: var(--stone) !important;
        color: var(--muted) !important;
        border: none;
        border-bottom: 1px solid var(--line) !important;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrameResizable"] [role="gridcell"] {
        background: var(--paper);
        color: var(--ink);
        border: none;
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
    }

    [data-testid="stDivider"] hr {
        border-color: var(--line);
    }

    [data-testid="stExpander"] {
        border: none;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: transparent;
        color: var(--ink) !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: var(--ink) !important;
    }

    [data-testid="stButton"] button[kind="primary"] {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_recommender():
    return MovieRecommender()


recommender = load_recommender()

def style_ranked_results(results):
    def alternating_rows(row):
        tint = "#fbfaf7" if row.name % 2 == 0 else "#f0ede6"
        return [f"background-color: {tint}; border: none"] * len(row)

    return results.style.apply(alternating_rows, axis=1).set_properties(
        **{"border": "none", "padding": "0.72rem 0.65rem"}
    )

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

    if st.button("Get My Recommendations", type="primary", key="get_my_recommendations"):
        with st.spinner("Generating recommendations..."):
            results, strategy_used = recommender.recommend_for_new_user(
                user_ratings=user_ratings,
                selected_genres=selected_genres,
                n=top_n,
            )
        st.caption(f"**Active Strategy:** {strategy_used}")
        st.dataframe(
            style_ranked_results(results),
            use_container_width=True,
            hide_index=True,
        )

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

    if st.button("Find Similar Movies", type="primary", key="find_similar_movies"):
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
            st.dataframe(
                style_ranked_results(results),
                use_container_width=True,
                hide_index=True,
            )

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
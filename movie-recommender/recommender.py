import os
import pickle
import pandas as pd

# =====================================
# GET ABSOLUTE PATHS
# =====================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "svd_model.pkl"
)

RATINGS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ratings.csv"
)

MOVIES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "movies.csv"
)

# =====================================
# LOAD MODEL
# =====================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =====================================
# LOAD DATA
# =====================================

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

# =====================================
# RECOMMEND FUNCTION
# =====================================

def recommend(user_id, n=10):

    # Movies already watched

    watched_movies = ratings[
        ratings["userId"] == user_id
    ]["movieId"].tolist()

    # Candidate movies

    candidate_movies = movies[
        ~movies["movieId"].isin(
            watched_movies
        )
    ]

    predictions = []

    for movie_id in candidate_movies["movieId"]:

        pred = model.predict(
            uid=user_id,
            iid=movie_id
        )

        predictions.append(
            (
                movie_id,
                float(pred.est)
            )
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for movie_id, score in predictions[:n]:

        title = movies[
            movies["movieId"] == movie_id
        ]["title"].values[0]

        recommendations.append(
            (
                title,
                score
            )
        )

    return recommendations
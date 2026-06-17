import pickle
import pandas as pd

model = pickle.load(
    open("svd_model.pkl", "rb")
)

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")


def recommend(user_id, n=10):

    watched = ratings[
        ratings["userId"] == user_id
    ]["movieId"].tolist()

    candidates = movies[
        ~movies["movieId"].isin(watched)
    ]

    predictions = []

    for movie_id in candidates["movieId"]:

        pred = model.predict(
            user_id,
            movie_id
        )

        predictions.append(
            (movie_id, pred.est)
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    result = []

    for movie_id, score in predictions[:n]:

        title = movies[
            movies["movieId"] == movie_id
        ]["title"].values[0]

        result.append(
            (title, score)
        )

    return result
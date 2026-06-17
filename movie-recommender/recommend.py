import pandas as pd
from surprise import Dataset, Reader, SVD

# ====================================
# LOAD DATA
# ====================================

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# ====================================
# CREATE SURPRISE DATASET
# ====================================

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

# ====================================
# TRAIN MODEL
# ====================================

trainset = data.build_full_trainset()

model = SVD(
    n_factors=100,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

model.fit(trainset)

# ====================================
# DEBUG TEST
# ====================================

print("\n===== SINGLE PREDICTION TEST =====")

test_pred = model.predict(1, 296)

print(test_pred)

# ====================================
# SELECT USER
# ====================================

user_id = 1

# ====================================
# MOVIES ALREADY WATCHED
# ====================================

watched_movies = ratings[
    ratings["userId"] == user_id
]["movieId"].tolist()

# ====================================
# MOVIE POPULARITY
# ====================================

movie_stats = ratings.groupby(
    "movieId"
).agg(
    num_ratings=("rating", "count")
).reset_index()

popular_movies = movie_stats[
    movie_stats["num_ratings"] >= 50
]["movieId"]

# ====================================
# UNWATCHED CANDIDATES
# ====================================

candidate_movies = movies[
    (~movies["movieId"].isin(watched_movies))
    &
    (movies["movieId"].isin(popular_movies))
]["movieId"].tolist()

print("\nCandidate Movies:", len(candidate_movies))

# ====================================
# GENERATE PREDICTIONS
# ====================================

predictions = []

for movie_id in candidate_movies:

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

# ====================================
# SORT
# ====================================

predictions.sort(
    key=lambda x: x[1],
    reverse=True
)

# ====================================
# TOP 20 RECOMMENDATIONS
# ====================================

print("\n===== TOP 20 RECOMMENDATIONS =====\n")

for rank, (movie_id, score) in enumerate(
    predictions[:20],
    start=1
):

    title = movies[
        movies["movieId"] == movie_id
    ]["title"].values[0]

    print(
        f"{rank}. {title}"
        f"\n   Predicted Rating: {score:.4f}\n"
    )

# ====================================
# PREDICTION STATS
# ====================================

scores = [p[1] for p in predictions]

print("\n===== PREDICTION STATS =====")

print("Max Prediction:", round(max(scores), 4))
print("Min Prediction:", round(min(scores), 4))
print("Average Prediction:", round(sum(scores)/len(scores), 4))
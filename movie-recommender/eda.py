import pandas as pd

ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

print("\n===== DATASET OVERVIEW =====")

print("Total Ratings:", len(ratings))
print("Total Users:", ratings['userId'].nunique())
print("Total Movies:", ratings['movieId'].nunique())
print("Average Rating:", round(ratings['rating'].mean(),2))

print("\n===== RATING DISTRIBUTION =====")

print(ratings['rating'].value_counts().sort_index())

print("\n===== MOST ACTIVE USERS =====")

user_activity = ratings.groupby('userId')['rating'].count()

print(
    user_activity
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== MOST POPULAR MOVIES =====")

movie_counts = ratings.groupby(
    'movieId'
)['rating'].count()

popular = movie_counts.reset_index()

popular.columns = [
    'movieId',
    'num_ratings'
]

popular = popular.merge(
    movies,
    on='movieId'
)

print(
    popular
    .sort_values(
        'num_ratings',
        ascending=False
    )
    [['title','num_ratings']]
    .head(10)
)
print("\n===== DATASET SPARSITY =====")

num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()

actual_ratings = len(ratings)

possible_ratings = num_users * num_movies

sparsity = 1 - (
    actual_ratings / possible_ratings
)

print(
    f"Sparsity: {sparsity:.4%}"
)
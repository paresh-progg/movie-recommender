import pandas as pd
import pickle

from surprise import Dataset
from surprise import Reader
from surprise import SVD

ratings = pd.read_csv("data/ratings.csv")

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

trainset = data.build_full_trainset()

model = SVD(
    n_factors=100,
    n_epochs=30,
    random_state=42
)

model.fit(trainset)

pickle.dump(
    model,
    open("svd_model.pkl", "wb")
)

print("Model Saved Successfully!")
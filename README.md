# Movie Recommendation System

A movie recommendation system built using Collaborative Filtering and SVD (Singular Value Decomposition) on the MovieLens dataset.

## Features

- Personalized movie recommendations
- SVD-based recommendation engine
- RMSE evaluation
- User-specific recommendations
- Streamlit web application

## Dataset

MovieLens Latest Small Dataset

- 100,836 ratings
- 610 users
- 9,724 movies

## Tech Stack

- Python
- Pandas
- Scikit-Surprise
- Streamlit
- Git
- GitHub

## Project Workflow

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Model Training using SVD
4. Recommendation Generation
5. Model Evaluation
6. Streamlit Deployment

## Results

- RMSE: 0.8798
- Dataset Sparsity: ~98.3%

## Run Locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Future Improvements

- Movie posters using TMDB API
- Search movies by title
- Hybrid recommendation system
- Cloud deployment

## Live Demo

[Streamlit App](https://movie-recommender-onm4wydov3sjfn3k9dbb8t.streamlit.app/)
## Author

Paresh Sahoo

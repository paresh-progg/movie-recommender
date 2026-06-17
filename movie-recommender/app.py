import streamlit as st

from recommender import recommend

st.title(
    "Movie Recommendation System"
)

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    max_value=610,
    value=1
)

if st.button(
    "Get Recommendations"
):

    recommendations = recommend(
        user_id
    )

    st.subheader(
        "Recommended Movies"
    )

    for movie, score in recommendations:

        st.write(
            f"⭐ {movie} ({score:.2f})"
        )
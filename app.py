import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_excel("online_course_recommendation_v2.xlsx")

# Remove duplicate courses
df = df.drop_duplicates(
    subset=["course_name", "instructor", "difficulty_level"]
).reset_index(drop=True)

# Feature Engineering
df["content"] = (
    df["course_name"].astype(str) + " " +
    df["instructor"].astype(str) + " " +
    df["difficulty_level"].astype(str) + " " +
    df["certification_offered"].astype(str) + " " +
    df["study_material_available"].astype(str)
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])

# Model
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(tfidf_matrix)

# Recommendation Function
def recommend(course):

    if course not in df["course_name"].values:
        return pd.DataFrame()

    idx = df[df["course_name"] == course].index[0]

    distances, indices = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=6
    )

    recommendations = df.iloc[indices[0][1:]]

    recommendations = recommendations[
        ["course_name", "instructor", "difficulty_level"]
    ]

    recommendations = recommendations.drop_duplicates()

    return recommendations


# ---------------- Streamlit UI ----------------

st.title("📚 Online Course Recommendation System")

course = st.selectbox(
    "Select Course",
    sorted(df["course_name"].unique())
)

if st.button("Recommend"):

    result = recommend(course)

    if result.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader("Recommended Courses")
        st.dataframe(result)
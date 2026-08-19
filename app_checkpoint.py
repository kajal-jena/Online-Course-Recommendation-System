import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_excel("online_course_recommendation_v2.xlsx")

# Feature Engineering
df['content'] = (
    df['course_name'].astype(str) + " " +
    df['instructor'].astype(str) + " " +
    df['difficulty_level'].astype(str) + " " +
    df['certification_offered'].astype(str) + " " +
    df['study_material_available'].astype(str)
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Train Nearest Neighbors Model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

# Create course index
indices = pd.Series(df.index, index=df['course_name']).drop_duplicates()

# Recommendation Function
def recommend_courses(course_name):

    idx = indices[course_name]

    distances, neighbors = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=6
    )

    recommendations = df.iloc[
        neighbors[0][1:]
    ][['course_name', 'instructor', 'difficulty_level']]

    return recommendations


# ---------------- Streamlit UI ----------------

st.title("📚 Online Course Recommendation System")

st.write("Select a course to get similar course recommendations.")

selected_course = st.selectbox(
    "Choose a Course",
    sorted(df['course_name'].unique())
)

if st.button("Recommend"):

    result = recommend_courses(selected_course)

    st.subheader("Top 5 Recommended Courses")

    st.dataframe(result)
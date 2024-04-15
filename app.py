import streamlit as st
import pandas as pd
import pickle

# Load the ratings dataset
ratings_df = pd.read_csv(r"C:\Users\91628\Desktop\UI_Project\ratings_merged.csv")

# Load the trained model
model = pickle.load(open(r"C:\Users\91628\Desktop\UI_Project\model.pkl", 'rb'))

# Function to get top N recommendations for a user
def get_top_n_recommendations(user_id, model, ratings_df, n=10):
    # Get a list of all movie IDs
    all_movie_ids = ratings_df['movieId'].unique()
    
    # Exclude movies already rated by the user
    rated_movie_ids = ratings_df.loc[ratings_df['userId'] == user_id, 'movieId'].unique()
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # Predict ratings for unrated movies
    unrated_predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    # Sort the predictions by estimated rating in descending order
    top_n_predictions = sorted(unrated_predictions, key=lambda x: x.est, reverse=True)[:n]
    
    # Get movie titles corresponding to the top-N predicted movie IDs
    top_n_movie_titles = [ratings_df.loc[ratings_df['movieId'] == pred.iid, 'original_title'].iloc[0] for pred in top_n_predictions]
    
    return top_n_movie_titles

# Function to handle button click event
def get_recommendations(user_id):
    try:
        recommendations = get_top_n_recommendations(user_id, model, ratings_df)  # Get movie recommendations for the user
        return recommendations
    except Exception as e:
        st.error(str(e))

# Streamlit interface
st.title("Movie Recommendation System")

# User ID input
user_id = st.number_input("Enter User ID:", min_value=1, step=1)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id)
    if recommendations:
        st.subheader("Top Recommendations:")
        st.write(recommendations)

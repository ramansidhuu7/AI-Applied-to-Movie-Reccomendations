import streamlit as st
import pandas as pd
import pickle

# Load the ratings dataset
ratings_df = pd.read_csv(r"C:\Users\91628\Desktop\UI_Project\ratings_merged.csv")

# Load the trained model for movie-based recommendations
movie_model = pickle.load(open(r"C:\Users\91628\Desktop\UI_Project\model.pkl", 'rb'))

# Load the content dataframe and cosine similarity matrix for movie-based recommendations
content_df = pd.read_csv(r"C:\Users\91628\Desktop\UI_Project\content_df.csv")
with open(r"C:\Users\91628\Desktop\UI_Project\cosine_similarity1.pkl", 'rb') as f:
    cos_sim_movie = pickle.load(f)

# Function to predict top similar movies based on movie title
def predict_movies(title, similarity_weight=0.7, top_n=10):
    try:
        # Reset index of content_df DataFrame
        data = content_df.reset_index()
        
        # Get the index of the movie with the given title
        index_movie = data[data['original_title'] == title].index
        
        # Transpose cosine similarity matrix to get similarities for the given movie
        similarity = cos_sim_movie[index_movie].T
        
        # Create a DataFrame containing similarity scores
        sim_df = pd.DataFrame(similarity, columns=['similarity'])
        
        # Concatenate the similarity DataFrame with the data DataFrame
        final_df = pd.concat([data, sim_df], axis=1)
        
        # Calculate final score using similarity_weight
        final_df['final_score'] = final_df['score']*(1-similarity_weight) + final_df['similarity']*similarity_weight
        
        # Sort DataFrame based on final score in descending order and select top_n movies
        final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
        
        # Set 'original_title' as index
        final_df_sorted.set_index('original_title', inplace=True)
        
        # Remove unnecessary columns
        final_df_sorted.drop(columns=['score', 'final_score', 'weighted_average', 'bag_of_words', 'similarity'], inplace=True)
        
        return final_df_sorted
    except Exception as e:
        st.error(e)

# Function to get top N recommendations for a user
def get_top_n_recommendations(user_id, n=10):
    try:
        # Get a list of all movie IDs
        all_movie_ids = ratings_df['movieId'].unique()
        
        # Exclude movies already rated by the user
        rated_movie_ids = ratings_df.loc[ratings_df['userId'] == user_id, 'movieId'].unique()
        unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
        
        # Predict ratings for unrated movies
        unrated_predictions = [movie_model.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
        
        # Sort the predictions by estimated rating in descending order
        top_n_predictions = sorted(unrated_predictions, key=lambda x: x.est, reverse=True)[:n]
        
        # Get movie titles corresponding to the top-N predicted movie IDs
        top_n_movie_titles = [ratings_df.loc[ratings_df['movieId'] == pred.iid, 'original_title'].iloc[0] for pred in top_n_predictions]
        
        return top_n_movie_titles
    except Exception as e:
        st.error(str(e))

# Streamlit interface
st.title("Movie Recommendation System")

# User's choice: Movie Title or User ID
choice = st.radio("Choose your option:", ("Movie Title", "User ID"))

if choice == "Movie Title":
    # Movie title input for movie-based recommendations
    movie_title = st.text_input("Enter Movie Title:")
    
    # Similarity weight input for movie-based recommendations
    similarity_weight_movie = st.slider("Similarity Weight (0-1):", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Top N input for movie-based recommendations
    top_n_movie = st.number_input("Top N Movies:", min_value=1, step=1, value=10)
    
    # Button to get movie-based recommendations
    if st.button("Get Movie Recommendations"):
        movie_recommendations = predict_movies(movie_title, similarity_weight_movie, top_n_movie)
        if movie_recommendations is not None and not movie_recommendations.empty:
            st.subheader("Top Movie Recommendations:")
            st.write(movie_recommendations)
        else:
            st.error("No movie recommendations found. Please check your inputs.")

elif choice == "User ID":
    # User ID input for user-based recommendations
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    
    # Top N input for user-based recommendations
    top_n_user = st.number_input("Top N Movies:", min_value=1, step=1, value=10)
    
    # Button to get user-based recommendations
    if st.button("Get User Recommendations"):
        user_recommendations = get_top_n_recommendations(user_id, top_n_user)
        if user_recommendations:
            st.subheader("Top User Recommendations:")
            st.write(user_recommendations)
        else:
            st.error("No user recommendations found. Please check your inputs.")

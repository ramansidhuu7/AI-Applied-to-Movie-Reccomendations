import streamlit as st
import pandas as pd
import pickle

# Function to predict top similar movies
def predict(title, similarity_weight=0.7, top_n=10):
    try:
        # Load the content dataframe
        content_df = pd.read_csv(r"C:\Users\91628\Desktop\UI_Project\content_df.csv")
        
        # Load the cosine similarity matrix
        with open(r"C:\Users\91628\Desktop\UI_Project\cosine_similarity1.pkl", 'rb') as f:
            cos_sim = pickle.load(f)
        
        # Reset index of content_df DataFrame
        data = content_df.reset_index()
        
        # Get the index of the movie with the given title
        index_movie = data[data['original_title'] == title].index
        
        # Transpose cosine similarity matrix to get similarities for the given movie
        similarity = cos_sim[index_movie].T
        
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
        
        # Remove the 'score' column
        final_df_sorted.drop(columns=['score','final_score','weighted_average','bag_of_words'], inplace=True)
        
        return final_df_sorted
    except Exception as e:
        st.error(e)

# Streamlit interface
st.title("Movie Recommendation System")

# Movie title input
movie_title = st.text_input("Enter Movie Title:")

# Similarity weight input
similarity_weight = st.slider("Similarity Weight (0-1):", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Top N input
top_n = st.number_input("Top N Movies:", min_value=1, step=1, value=10)

# Button to get recommendations
if st.button("Get Recommendations"):
    recommendations = predict(movie_title, similarity_weight, top_n)
    if recommendations is not None and not recommendations.empty:
        st.subheader("Top Recommendations:")
        st.write(recommendations)
    else:
        st.error("No recommendations found. Please check your inputs.")

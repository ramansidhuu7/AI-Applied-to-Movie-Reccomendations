from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Define the directory path
directory = "C:\\Users\\91628\\Desktop\\AI Project\\User_interface\\movie_recommendation_system"

# Load the predict function from the pickle file
with open(os.path.join(directory, 'predict_function.pkl'), 'rb') as file:
    movie_recommender = pickle.load(file)

# Load the cosine similarity scores from the pickle file
with open(os.path.join(directory, 'cosine_similarity.pkl'), 'rb') as file:
    cosine_similarity = pickle.load(file)

@app.route('/')
def index():
    return render_template(os.path.join(directory, 'templates', 'index.html'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        title = request.form['movie_title']
        similarity_weight = float(request.form['similarity_weight'])
        top_n = int(request.form['top_n'])

        # Predict top similar movies using the loaded recommendation logic
        result = movie_recommender.predict(title, cosine_similarity, similarity_weight, top_n)
        
        return render_template(os.path.join(directory, 'templates', 'result.html'), result=result.to_html())
    except Exception as e:
        return render_template(os.path.join(directory, 'templates', 'error.html'), error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

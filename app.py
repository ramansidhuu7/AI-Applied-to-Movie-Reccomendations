from flask import Flask, redirect

app = Flask(__name__)

@app.route('/')
def index():
    return redirect('https://movie-recommendation-system.s3.amazonaws.com/index.html')

if __name__ == '__main__':
    app.run(debug=True)

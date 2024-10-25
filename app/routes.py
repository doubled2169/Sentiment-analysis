from flask import render_template, request, jsonify
from app import app
import joblib

# Load model and TF-IDF vectorizer
model = joblib.load('models/sentiment_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        review_text = request.form['review_text']
        
        # Preprocess and transform the text using the TF-IDF vectorizer
        transformed_text = tfidf.transform([review_text])
        
        # Predict sentiment using the trained model
        prediction = model.predict(transformed_text)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return render_template('index.html', review=review_text, sentiment=sentiment)

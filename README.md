# Sentiment-analysis
Objective: To build a system that can classify product reviews (or any text) as positive, negative, or neutral based on the sentiment of the text. This could be extended to more complex classifications like identifying emotions (happy, sad, angry) or even detecting fake reviews.

# Sentiment Analysis of Product Reviews

This is a Flask web application that performs sentiment analysis on product reviews. Users can input a product review, and the model will predict whether the sentiment is positive or negative.

## Features
- Input a product review through a web form.
- See the predicted sentiment (positive or negative).
- Built with Python, Flask, and Scikit-learn.

## Requirements
- Python 3.x
- Install dependencies: `pip install -r requirements.txt`

## How to Run the App Locally
1. Clone the repository: `git clone <repo_url>`
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Flask app: `python app.py`
4. Open a browser and go to `http://localhost:5000`

## How to Deploy to Heroku
1. Install Heroku CLI.
2. Run the command `heroku create` to create a new app.
3. Deploy the app using `git push heroku main`.


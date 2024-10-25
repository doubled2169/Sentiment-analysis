import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

nltk.data.path.append("path to nltk data")

# Load dataset
reviews_df = pd.read_csv('data/amazon_reviews.csv')

# Create a sentiment column based on the Score
# Assuming Score 1-2 is negative, 4-5 is positive, and 3 is neutral
reviews_df['sentiment'] = reviews_df['Score'].apply(lambda x: 1 if x > 3 else (0 if x == 3 else -1))

# Preprocess text function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Apply preprocessing to the reviews
reviews_df['cleaned_review'] = reviews_df['Text'].apply(preprocess_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(reviews_df['cleaned_review'])
y = reviews_df['sentiment']  # Target variable is now 'sentiment'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model and vectorizer
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

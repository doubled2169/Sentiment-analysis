import pandas as pd

# Load your dataset
reviews_df = pd.read_csv('data/amazon_reviews.csv')

# Print the column names
print(reviews_df.columns)
print(reviews_df.shape)


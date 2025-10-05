import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = "D:\\financial Sentiment Analysis for Market Prediction\\data.csv"
data = pd.read_csv("D:\\financial Sentiment Analysis for Market Prediction\\data.csv")

# Clean column names to avoid errors
data.columns = data.columns.str.strip()

# Define a preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'Sentence' column
data['Processed_Sentence'] = data['Sentence'].apply(preprocess_text)

# Display the first few rows of the updated dataset
print(data[['Sentence', 'Processed_Sentence', 'Sentiment']].head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Vectorize the processed text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for performance
X = vectorizer.fit_transform(data['Processed_Sentence'])

# Map sentiment labels to numeric values
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
y = data['Sentiment'].map(sentiment_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# Simulate dates and stock prices for demonstration
np.random.seed(42)
data['Date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='H')  # Simulated timestamps
data['Stock_Price'] = np.random.uniform(100, 200, len(data))  # Simulated stock prices

# Aggregate sentiment scores by day
data['Sentiment_Score'] = data['Sentiment'].map(sentiment_mapping)
daily_sentiment = data.groupby(data['Date'].dt.date)['Sentiment_Score'].mean().reset_index()
daily_sentiment.columns = ['Date', 'Average_Sentiment']

# Merge with stock price data
daily_stock_prices = data.groupby(data['Date'].dt.date)['Stock_Price'].mean().reset_index()
daily_stock_prices.columns = ['Date', 'Average_Stock_Price']

# Combine sentiment and stock price data
time_series_data = pd.merge(daily_sentiment, daily_stock_prices, on='Date')

# Visualize the relationship
plt.figure(figsize=(10, 5))
plt.plot(time_series_data['Date'], time_series_data['Average_Sentiment'], label='Average Sentiment', color='blue')
plt.plot(time_series_data['Date'], time_series_data['Average_Stock_Price'], label='Average Stock Price', color='green')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Sentiment vs. Stock Price')
plt.show()

# Build a regression model to predict stock price based on sentiment
X = time_series_data['Average_Sentiment'].values.reshape(-1, 1)
y = time_series_data['Average_Stock_Price'].values
reg_model = LinearRegression()
reg_model.fit(X, y)

# Predict and evaluate
y_pred = reg_model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Visualize predictions
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, y_pred, color='red', label='Predicted Prices')
plt.xlabel('Average Sentiment')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Sentiment vs. Predicted Stock Price')
plt.show()



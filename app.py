from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize Flask application
app = Flask(__name__)

# Load your trained model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') # Load your saved vectorizer

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cleaning and Preprocessing Tweet
def clean_and_preprocess_tweet(tweet):
    # Cleaning
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'\d+', '', tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stopwords.words('english')]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    # Preprocessing complete
    return ' '.join(stemmed_words)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tweet = request.form['tweet']

        # Clean and preprocess the tweet
        processed_tweet = clean_and_preprocess_tweet(tweet)

        # Transform the tweet using the loaded TF-IDF vectorizer
        vectorized_tweet = tfidf_vectorizer.transform([processed_tweet])

        # Predict the class
        prediction = model.predict(vectorized_tweet)

        # Convert prediction to a readable format
        result = 'Class of the Tweet: ' + str(prediction[0])
        return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)

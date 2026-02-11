# https://medium.com/@zeeshankhan0094/how-to-make-sentiment-analysis-model-with-python-a-practical-guide-4c633880e295

import nltk
import pandas as pd
#from nltk.corpus import twitter_samples
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

twitter_samples = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)

#positive_tweets = [(t, "pos") for t in twitter_samples.strings("positive_tweets.json")]
#negative_tweets = [(t, "neg") for t in twitter_samples.strings("negative_tweets.json")]

# Combine positive and negative tweets
#tweets = positive_tweets + negative_tweets
tweets = []
for idx, row in twitter_samples.iterrows():
    if row[0] == 4:
        sentiment = "pos"
    elif row[0] == 2:
        sentiment = "neutral"
    else:
        sentiment = "neg"
    text = row[5]  # Column 5: tweet text
    tweets.append((text, sentiment))

# Shuffle the tweets
import random
random.shuffle(tweets)

keywords = {"organic", "natural", "egg", "milk", "dairy", "meat", "gmo", "genetically modified", "egg", "livestock", "free-range", "cage-free", "humane"}

def extract_features(words):
    return dict([(word, True) for word in words])

# Define a function to preprocess the data
def preprocess_data(tweets):
    stop_words = set(stopwords.words('english'))
    processed_tweets = []
    #print(tweets)
    
    tokenized_words = []
    for words, sentiment in tweets:
        tokenized_words.append((words.split(), sentiment))
    #print(tokenized_words)

    for words, sentiment in tokenized_words:
        # Remove stopwords and convert to lowercase
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        #print(words)
        
        processed_tweets.append((words, sentiment))
    
    #print(processed_tweets)
    
    keyword_tweets = []

    for words, sentiment in processed_tweets:
        # Only take in tweets that contain keywords
        if any(k in words for k in keywords):
            keyword_tweets.append((words, sentiment))
    
    #print(keyword_tweets)

    print(f'Processed Tweets: {len(keyword_tweets)}')

    return keyword_tweets

# Preprocess the data
processed_tweets = preprocess_data(tweets)

# Split the data into training and testing sets
split_ratio = 0.8
split = int(len(processed_tweets) * split_ratio)
train_data, test_data = processed_tweets[:split], processed_tweets[split:]
print(f'Training Data: {len(train_data)}')
print(f'Testing Data: {len(test_data)}')


# Extract features from the training data
training_features = [(extract_features(words), sentiment) for words, sentiment in train_data]

# Train the Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(training_features)

# Test the classifier on the testing data
test_features = [(extract_features(words), sentiment) for words, sentiment in test_data]
accuracy = nltk_accuracy(classifier, test_features)

print(f'Accuracy: {accuracy:.2%}')

from nltk.metrics import ConfusionMatrix

# Test the classifier on the testing data
test_features = [(extract_features(words), sentiment) for words, sentiment in test_data]
test_labels = [sentiment for _, sentiment in test_data]

# Get predictions
predicted_labels = [classifier.classify(features) for features, _ in test_features]

# Calculate and print confusion matrix
cm = ConfusionMatrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Calculate and print additional metrics
precision = cm['pos', 'pos'] / (cm['pos', 'pos'] + cm['neg', 'pos'])
recall = cm['pos', 'pos'] / (cm['pos', 'pos'] + cm['pos', 'neg'])
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision:.2%}')
print(f'Recall: {recall:.2%}')
print(f'F1 Score: {f1_score:.2%}')

# Examples without labels
examples = [
    "This movie is fantastic! I loved every moment of it.",
    "The cinematography and acting were outstanding. A must-watch!",
    "The plot was confusing, and the characters were poorly developed.",
    "I regret watching this movie. It was a waste of time.",
    "I regret watching this. It was a waste of money."
]

# Function to predict sentiment
def predict_sentiment(text):
    words = word_tokenize(text)
    features = extract_features(words)
    sentiment = classifier.classify(features)
    return sentiment

# Make predictions and display results
predictions = []
for example in examples:
    prediction = predict_sentiment(example)
    predictions.append(prediction)
    print(f'Example Prediction: {prediction}')

    import matplotlib.pyplot as plt

# Visualize sentiment distribution
sentiment_counts = {'pos': 0, 'neutral': 0, 'neg': 0}
for prediction in predictions:
    if prediction in sentiment_counts:
        sentiment_counts[prediction] += 1

labels = list(sentiment_counts.keys())
counts = list(sentiment_counts.values())

plt.bar(labels, counts, color=['green', 'gray', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Predicted Sentiment Distribution for Examples')
plt.show()
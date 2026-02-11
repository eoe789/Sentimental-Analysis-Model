import pandas as pd
import re
from datetime import datetime, timedelta
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('stopwords')

df = pd.read_csv('reddit_meat.csv')
# Filtering Start

# Filter Automod
no_automod = []
for index, row in df.iterrows():
    author = row['author']
    if author == "AutoModerator":
        continue

    no_automod.append(row)

no_automod_df = pd.DataFrame(no_automod)

#no_automod_df.to_csv("reddit_data_no_automod.csv")

# Time Filter
#year_filter_df = no_automod_df
no_automod_df['created_utc'] = pd.to_datetime(no_automod_df['created_utc'])

# Calculate date boundaries
# three_years_ago = pd.Timestamp("2022-01-01", tz="UTC")
# one_year_ago = pd.Timestamp("2026-01-01", tz="UTC")

# year_filter_df = no_automod_df[
#     (pd.to_datetime(no_automod_df['created_utc'], unit='s', utc=True) <= three_years_ago) &
#     (pd.to_datetime(no_automod_df['created_utc'], unit='s', utc=True) >= one_year_ago)
# ]
year_filter_df = no_automod_df.copy()

# Language Filtering (also removes empty text posts)
def is_english(text: str) -> bool:
    if pd.isna(text):
        return False
    text = str(text).strip()

    if len(text.split()) < 3:
        return False
    
    if not text:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


language_filter = []
for index, row in year_filter_df.iterrows():
    text = row.get("selftext", "")
    if not is_english(text):
        continue

    language_filter.append(row)

language_filter_df = pd.DataFrame(language_filter)
#language_filter_df.to_csv("reddit_english.csv")

final_df = pd.DataFrame(['raw_text', 'filtered_text', 'tokens'])
final_df['raw_text'] = language_filter_df['selftext']

# Remove links and users
text_cleanup = language_filter_df.copy()
for index, row in text_cleanup.iterrows():
    text = str(row['selftext']).split()

    cleaned_words = []
    for word in text:
        if word.startswith('http') or word.startswith('u/'):
            continue
        cleaned_words.append(word)

    row['selftext'] = ' '.join(cleaned_words)
    text_cleanup.at[index, 'selftext'] = row['selftext']

# Remove stop words, lemmatize and convert to lowercase
stop_words = set(stopwords.words('english'))
no_stop_words = text_cleanup.copy()

lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

for index, row in no_stop_words.iterrows():
    words = str(row['selftext']).split()

    words = [
        word.lower()
        for word in words
        if re.search(r"[a-zA-Z]", word)
        and word.lower() not in stop_words
    ]

    # POS tagging
    tagged_words = nltk.pos_tag(words)

    # Lemmatization
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_words
    ]

    row['selftext'] = ' '.join(lemmatized_words)
    no_stop_words.at[index, 'selftext'] = row['selftext']

# Deduplication
deduplication = no_stop_words.drop_duplicates(subset=['selftext'])

# Length Filtering
length_filter = []
for index, row in deduplication.iterrows():
    text = row['selftext']
    text = text.split()
    if len(text) < 5:
        continue

    length_filter.append(row)

length_filter_df = pd.DataFrame(length_filter)

final_df['filtered_text'] = length_filter_df['selftext']
final_df['tokens'] = final_df['filtered_text'].apply(lambda x: str(x).split() if pd.notna(x) else [])

output_df = final_df[['raw_text', 'filtered_text', 'tokens']]

output_df.to_csv('reddit_processed_text.csv', index=False)

attrition = {
    "raw": len(df),
    "no_automod": len(no_automod_df),
    "date_filtered": len(year_filter_df),
    "english": len(language_filter_df),
    "deduplicated": len(deduplication),
    "length_filtered": len(length_filter_df)
}

print(attrition)

# 1
# Track each query
# Verify at least 80% relavant
# Only track posts from 1 to 3 years ago
# Track engagement metrics
# Anonymize users

# 2
# Text Cleanup
# Emoji Hanlding
# Lamenization
# Remove duplicate posts
# Filter for only English posts
# Attrition table
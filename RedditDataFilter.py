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

SUBREDDIT = 'HealthyFood'

df = pd.read_csv(f'reddit_{SUBREDDIT}.csv')
# Filtering Start

#df.to_csv("test.csv")

count = 0

# Filter Automod
no_automod = []
for index, row in df.iterrows():
    author = row['author']
    if author == "AutoModerator":
        continue
    no_automod.append(row)

no_automod_df = pd.DataFrame(no_automod)
#no_automod_df.to_csv("test.csv")

# Time Filter
no_automod_df['created_utc'] = pd.to_datetime(no_automod_df['created_utc'], unit='s', utc=True)

# Calculate date boundaries
start_date = pd.Timestamp("2018-01-01", tz="UTC")
end_date = pd.Timestamp.utcnow()

year_filter = no_automod_df[
    (no_automod_df['created_utc'] >= start_date) &
    (no_automod_df['created_utc'] <= end_date)
]

year_filter_df = pd.DataFrame(year_filter)
#year_filter_df.to_csv("test.csv")

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
    title_text = row.get("title", "")
    body_text = row.get("selftext", "")
    text = f"{title_text} {body_text}".strip()
    if not is_english(text):
        continue

    language_filter.append(row)

language_filter_df = pd.DataFrame(language_filter)
language_filter_df['combined_text'] = (
    language_filter_df.get('title', '').fillna('')
    + ' '
    + language_filter_df.get('selftext', '').fillna('')
).str.strip()
language_filter_df['raw_text'] = language_filter_df['combined_text']
#language_filter_df.to_csv("test.csv")

# Remove links and users
text_cleanup = language_filter_df.copy()
for index, row in text_cleanup.iterrows():
    text = str(row['combined_text']).split()

    cleaned_words = []
    for word in text:
        if word.startswith('http') or word.startswith('u/'):
            continue
        cleaned_words.append(word)

    row['combined_text'] = ' '.join(cleaned_words)
    text_cleanup.at[index, 'combined_text'] = row['combined_text']

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
    words = str(row['combined_text']).split()

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

    row['combined_text'] = ' '.join(lemmatized_words)
    no_stop_words.at[index, 'combined_text'] = row['combined_text']

no_stop_words_df = pd.DataFrame(no_stop_words)
no_stop_words_df.to_csv("test.csv")

# Deduplication
deduplication = no_stop_words.drop_duplicates(subset=['combined_text'])
deduplication_df = pd.DataFrame(deduplication)
#deduplication_df.to_csv("test.csv")

# Length Filtering
length_filter = []
for index, row in deduplication_df.iterrows():
    text = row['combined_text']
    text = text.split()
    if len(text) < 5:
        continue

    length_filter.append(row)

length_filter_df = pd.DataFrame(length_filter)

length_filter_df.to_csv('result.csv')

# Key Word Filtering

# keywords = {"organic", "natural", "egg", "milk", "dairy", "meat", "beef", "chicken", "gmo", "egg", "livestock"}

# keyword_filter = []
# for index, row in length_filter_df.iterrows():
#     text = str(row['combined_text']).lower()
#     words = text.split()

#     has_match = False
#     for keyword in keywords:
#         keyword_lower = keyword.lower()
#         if ' ' in keyword_lower:
#             if keyword_lower in text:
#                 has_match = True
#                 break
#         elif keyword_lower in words:
#             has_match = True
#             break

#     if has_match:
#         keyword_filter.append(row)
#
#keyword_filter_df = pd.DataFrame(keyword_filter)

final_df = length_filter_df[['raw_text']].copy()
final_df['filtered_text'] = length_filter_df['combined_text']
final_df['tokens'] = final_df['filtered_text'].apply(lambda x: str(x).split() if pd.notna(x) else [])

output_df = final_df[['raw_text', 'filtered_text', 'tokens']]

output_df.to_csv(f'reddit_{SUBREDDIT}_filtered.csv', index=False)

attrition = {
    "raw": len(df),
    "no_automod": len(no_automod_df),
    "date_filtered": len(year_filter_df),
    "english": len(language_filter_df),
    "stop-words": len(no_stop_words_df),
    "deduplicated": len(deduplication_df),
    "length_filtered": len(length_filter_df),
    #"key_word": len(keyword_filter_df)
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
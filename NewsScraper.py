import feedparser
import time
import pandas as pd
import re
import html
from urllib.parse import quote_plus
from tqdm import tqdm
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Configuration - same keywords as DataFilter.py
class config:
    SEARCH_QUERIES = {
        "organic",
        "natural",
        "egg",
        "milk",
        "dairy",
        "meat",
        "beef",
        "chicken",
        "gmo",
        "genetically modified",
        "livestock",
    }
    MAX_ITEMS_PER_QUERY = 100

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    cleaned = html.unescape(str(text))
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def is_english(text: str) -> bool:
    if not text:
        return False
    text = str(text).strip()
    if len(text.split()) < 3:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def scrape_news_rss(queries=None, max_per_query=config.MAX_ITEMS_PER_QUERY):
    """Scrape news articles via Google News RSS (no API key)."""
    queries = queries or sorted(config.SEARCH_QUERIES)
    rows = []
    for q in tqdm(queries, desc="News RSS"):
        url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            feed = feedparser.parse(
                url,
                request_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"},
            )
            for i, entry in enumerate(feed.entries):
                if i >= max_per_query:
                    break
                title = clean_text(entry.get("title", ""))
                summary = clean_text(entry.get("summary", ""))
                text = f"{title}. {summary}" if summary else title
                if len(text) < 20:
                    continue
                rows.append({
                    "source": "news_rss",
                    "query": q,
                    "text": text,
                    "title": title,
                    "url": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            tqdm.write(f"News RSS error for '{q}': {e}")
        time.sleep(0.5)
    return rows

def process_articles(articles):
    """Apply filtering pipeline: language, stop-words, lemmatization."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed = []
    
    for article in tqdm(articles, desc="Processing"):
        combined_text = article.get("text", "")
        
        # Language filter
        if not is_english(combined_text):
            continue
        
        # Stop-word removal and lemmatization
        words = str(combined_text).split()
        words = [
            word.lower()
            for word in words
            if re.search(r"[a-zA-Z]", word)
            and word.lower() not in stop_words
        ]
        
        # POS tagging and lemmatization
        tagged_words = nltk.pos_tag(words)
        lemmatized_words = [
            lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in tagged_words
        ]
        
        filtered_text = " ".join(lemmatized_words)
        if len(filtered_text) < 5:
            continue
        
        tokens = filtered_text.split() if filtered_text else []
        
        processed.append({
            "query": article.get("query", ""),
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "published": article.get("published", ""),
            "raw_text": combined_text,
            "filtered_text": filtered_text,
            "tokens": tokens,
        })
    
    return processed

# Run the scraper
articles = scrape_news_rss()
print(f"Raw articles scraped: {len(articles)}")

# Process with filtering pipeline
processed = process_articles(articles)
print(f"After language/lemmatization: {len(processed)}")

# Deduplication by filtered_text
deduplicated = []
seen_text = set()
for item in processed:
    filtered = item["filtered_text"]
    if filtered not in seen_text:
        seen_text.add(filtered)
        deduplicated.append(item)

print(f"After deduplication: {len(deduplicated)}")

# Length filter: keep only posts with >= 5 tokens
length_filtered = [item for item in deduplicated if len(item["tokens"]) >= 5]
print(f"After length filter (>= 5 tokens): {len(length_filtered)}")

# Save to CSV
df = pd.DataFrame(length_filtered)
df.to_csv('google_news_filtered.csv', index=False)

print(f"✓ Saved {len(length_filtered)} articles → google_news_filtered.csv")

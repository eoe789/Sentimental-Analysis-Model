import ast
import pandas as pd
import torch
from collections import Counter
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SUBREDDIT = 'nutrition'
INPUT_CSV = f"combined_raw_filtered_tokens.csv"
TOKENS_COLUMN = "tokens"
OUTPUT_CSV = f"total_sentiment_score.csv"
CLUSTER_TEXT_COLUMN = "filtered_text"
NUM_CLUSTERS = 10
CLUSTER_SUMMARY_CSV = "cluster_summary.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

df = pd.read_csv(INPUT_CSV)

def tokens_to_text(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, list):
        return " ".join(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return " ".join(parsed)
        except (ValueError, SyntaxError):
            pass
        return value
    return str(value)

texts = df[TOKENS_COLUMN].apply(tokens_to_text).fillna("").tolist()

batch_size = 16
all_scores = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        all_scores.extend(probs.tolist())

labels = ["negative", "neutral", "positive"]
score_df = pd.DataFrame(all_scores, columns=[f"score_{l}" for l in labels])
score_df["predicted_label"] = score_df.idxmax(axis=1).str.replace("score_", "", regex=False)

label_counts = score_df["predicted_label"].value_counts()
print(f"Negative: {label_counts.get('negative', 0)}")
print(f"Neutral: {label_counts.get('neutral', 0)}")
print(f"Positive: {label_counts.get('positive', 0)}")

output_df = pd.concat([df, score_df], axis=1)
output_df.to_csv(OUTPUT_CSV, index=False)

if CLUSTER_TEXT_COLUMN not in output_df.columns:
    output_df[CLUSTER_TEXT_COLUMN] = output_df[TOKENS_COLUMN].apply(tokens_to_text)


def add_text_clusters(
    df: pd.DataFrame,
    text_col: str = "filtered_text",
    n_clusters: int = 10,
    random_state: int = 42,
    max_features: int = 8000,
    top_terms_per_cluster: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("Clustering skipped: scikit-learn is not installed.")
        print("Install with: python -m pip install scikit-learn")
        return df, None

    if text_col not in df.columns:
        print(f"Clustering skipped: column '{text_col}' not found.")
        return df, None

    result_df = df.copy()
    texts = result_df[text_col].fillna("").astype(str)
    valid_mask = texts.str.strip() != ""

    if valid_mask.sum() < 2:
        print("Clustering skipped: not enough non-empty text rows.")
        return result_df, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
    )
    tfidf_matrix = vectorizer.fit_transform(texts[valid_mask])

    effective_clusters = min(n_clusters, tfidf_matrix.shape[0])
    if effective_clusters < 2:
        print("Clustering skipped: effective cluster count is less than 2.")
        return result_df, None

    kmeans = KMeans(n_clusters=effective_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    result_df["cluster_id"] = -1
    result_df.loc[valid_mask, "cluster_id"] = cluster_labels

    feature_names = np.array(vectorizer.get_feature_names_out())
    summary_rows = []
    for cluster_id in range(effective_clusters):
        centroid = kmeans.cluster_centers_[cluster_id]
        top_indices = np.argsort(centroid)[::-1][:top_terms_per_cluster]
        top_terms = ", ".join(feature_names[top_indices])
        size = int((cluster_labels == cluster_id).sum())
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "top_terms": top_terms,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("size", ascending=False)
    return result_df, summary_df


def compute_semantic_drivers(
    df: pd.DataFrame,
    tokens_col: str = "tokens",
    label_col: str = "predicted_label",
    alpha: float = 0.5,
    top_n: int = 20
) -> dict:
    """
    Compute semantic drivers of sentiment using log-odds ratio with Dirichlet prior.
    
    Returns dict with 'negative' and 'positive' keys containing ranked driver words.
    """
    # Parse tokens for each row
    def parse_tokens(value):
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
            # Fallback: split by whitespace
            return value.split()
        return []
    
    df_copy = df.copy()
    df_copy['parsed_tokens'] = df_copy[tokens_col].apply(parse_tokens)
    
    # Count word occurrences per sentiment
    word_counts = {
        'negative': Counter(),
        'neutral': Counter(),
        'positive': Counter()
    }
    
    for _, row in df_copy.iterrows():
        label = row[label_col]
        tokens = row['parsed_tokens']
        if label in word_counts:
            word_counts[label].update(tokens)
    
    # Get all unique words
    all_words = set()
    for counter in word_counts.values():
        all_words.update(counter.keys())
    
    # Compute total counts per sentiment
    total_counts = {
        label: sum(counter.values()) for label, counter in word_counts.items()
    }
    
    # Compute log-odds ratio for negative vs positive
    vocab_size = len(all_words)
    log_odds = {}
    
    for word in all_words:
        # P(word|negative) with Dirichlet prior
        count_neg = word_counts['negative'][word]
        p_word_neg = (count_neg + alpha) / (total_counts['negative'] + alpha * vocab_size)
        
        # P(word|positive) with Dirichlet prior
        count_pos = word_counts['positive'][word]
        p_word_pos = (count_pos + alpha) / (total_counts['positive'] + alpha * vocab_size)
        
        # Log-odds ratio
        log_odds[word] = np.log(p_word_neg / p_word_pos)
    
    # Sort by log-odds
    sorted_words = sorted(log_odds.items(), key=lambda x: x[1], reverse=True)
    
    # Top negative drivers (highest log-odds favoring negative)
    negative_drivers = [(word, log_odds[word], word_counts['negative'][word]) 
                       for word, _ in sorted_words[:top_n]]
    
    # Top positive drivers (lowest log-odds favoring positive)
    positive_drivers = [(word, log_odds[word], word_counts['positive'][word]) 
                       for word, _ in sorted_words[-top_n:][::-1]]
    
    return {
        'negative': negative_drivers,
        'positive': positive_drivers,
        'word_counts': word_counts,
        'total_counts': total_counts
    }


def print_semantic_drivers(drivers: dict, top_n: int = 10):
    """
    Print ranked semantic drivers in a formatted table.
    """
    print("\n" + "="*80)
    print("SEMANTIC DRIVERS OF SENTIMENT")
    print("="*80)
    
    print(f"\n{'Rank':<6}{'Negative Drivers':<30}{'Log-Odds':<15}{'Count':<10}")
    print("-" * 80)
    for i, (word, log_odds_val, count) in enumerate(drivers['negative'][:top_n], 1):
        print(f"{i:<6}{word:<30}{log_odds_val:>12.4f}{count:>13}")
    
    print(f"\n{'Rank':<6}{'Positive Drivers':<30}{'Log-Odds':<15}{'Count':<10}")
    print("-" * 80)
    for i, (word, log_odds_val, count) in enumerate(drivers['positive'][:top_n], 1):
        print(f"{i:<6}{word:<30}{log_odds_val:>12.4f}{count:>13}")
    
    print("\n" + "="*80)
    print("EXPLANATION:")
    print("Log-odds > 0: word occurs more frequently in negative sentiment")
    print("Log-odds < 0: word occurs more frequently in positive sentiment")
    print("Higher absolute values indicate stronger sentiment drivers")
    print("="*80 + "\n")


# Compute and display semantic drivers
print("\nComputing semantic drivers...")
drivers = compute_semantic_drivers(output_df, TOKENS_COLUMN, "predicted_label")
print_semantic_drivers(drivers, top_n=10)

print("Computing text clusters...")
output_df, cluster_summary_df = add_text_clusters(
    output_df,
    text_col=CLUSTER_TEXT_COLUMN,
    n_clusters=NUM_CLUSTERS,
)

output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved sentiment + clusters to: {OUTPUT_CSV}")

if cluster_summary_df is not None:
    cluster_summary_df.to_csv(CLUSTER_SUMMARY_CSV, index=False)
    print(f"Saved cluster summary to: {CLUSTER_SUMMARY_CSV}")
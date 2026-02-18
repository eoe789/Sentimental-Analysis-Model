import ast
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SUBREDDIT = 'nutrition'
INPUT_CSV = f"reddit_{SUBREDDIT}_filtered.csv"
TOKENS_COLUMN = "tokens"
OUTPUT_CSV = f"reddit_{SUBREDDIT}_sentiment_score.csv"

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
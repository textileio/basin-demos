import torch
from polars import DataFrame, Series
from transformers import pipeline

# Detect the appropriate device for pipeline processing
if torch.backends.mps.is_available():
    device_type = "mps"
elif torch.cuda.is_available():
    device_type = "cuda"
else:
    device_type = "cpu"

# Models the will be used to classify the data
MODEL_CLASSIFIER = "facebook/bart-large-mnli"
MODEL_SENTIMENT = "nlptown/bert-base-multilingual-uncased-sentiment"

# Initialize the models
classifier = pipeline(
    "zero-shot-classification", model=MODEL_CLASSIFIER, device=device_type
)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=MODEL_SENTIMENT,
    device=device_type,
)

# Define the labels for topic classification
labels = [
    "technology",
    "politics",
    "entertainment",
    "health",
    "education",
    "finance",
    "sports",
    "other",
]

# Redefine `nlptown`'s sentiment labels (e.g., "1 star" -> "negative")
sentiment_labels = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


# Attach labels and sentiment to each row from the dataset
def process_row(text: str) -> tuple[str, str]:
    # Topic classification
    classification_result = classifier(text, candidate_labels=labels)
    topic = classification_result["labels"][0]  # Take the top label

    # Sentiment analysis
    sentiment_result = sentiment_analyzer(text)
    sentiment_stars = sentiment_result[0]["label"]
    sentiment = sentiment_labels[sentiment_stars]

    return topic, sentiment


def process_df(df: DataFrame) -> DataFrame:
    # Apply the processing function to each row
    processed_data = [process_row(row[3]) for row in df.iter_rows()]

    # Add the results back to the DataFrame
    topics, sentiments = zip(*processed_data)
    return df.with_columns([Series("topic", topics), Series("sentiment", sentiments)])

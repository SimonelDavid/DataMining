# sentiment_analysis.py

# Load extended vocabularies
with open("../topan/positive_words.txt", "r", encoding="utf-8") as pos_file:
    positive_words = pos_file.read().splitlines()

with open("../topan/negative_words.txt", "r", encoding="utf-8") as neg_file:
    negative_words = neg_file.read().splitlines()


def basic_sentiment_analysis(review, positive_words, negative_words):
    """
    Basic sentiment analysis: counts positive and negative words to determine tone.
    """
    review_lower = review.lower()  # Case-insensitive matching
    positive_count = sum(review_lower.count(word) for word in positive_words)
    negative_count = sum(review_lower.count(word) for word in negative_words)

    # Determine tone based on counts
    tone = "Positive" if positive_count > negative_count else "Negative" if negative_count > positive_count else "Neutral"

    return tone

import nltk
#nltk.download('all')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
class Lexicon:

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def preprocess_text(self,text):
        # Tokenize the text

        tokens = word_tokenize(text.lower())

        # Remove stop words

        filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

        # Lemmatize the tokens

        lemmatizer = WordNetLemmatizer()

        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        # Join the tokens back into a string

        processed_text = ' '.join(lemmatized_tokens)

        return processed_text

    def get_sentiment(self,text):
        scores = self.analyzer.polarity_scores(text)

        sentiment = 1 if scores['pos'] > 0 else 0

        return sentiment



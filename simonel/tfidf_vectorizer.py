import numpy as np
from collections import Counter
from math import log

class TfidfVectorizerCustom:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary = {}

    def fit_transform(self, corpus):
        """Learn the vocabulary and return term-document matrix."""
        doc_count = len(corpus)
        term_freq = Counter()
        doc_freq = Counter()

        # Tokenize and calculate term/document frequencies
        tokenized_corpus = []
        for doc in corpus:
            tokens = doc.lower().split()
            tokenized_corpus.append(tokens)
            term_freq.update(tokens)
            doc_freq.update(set(tokens))

        # Sort by frequency and keep the top max_features terms
        if self.max_features:
            most_common = [word for word, _ in term_freq.most_common(self.max_features)]
        else:
            most_common = list(term_freq.keys())

        self.vocabulary = {word: idx for idx, word in enumerate(most_common)}

        # Create TF-IDF matrix
        tfidf_matrix = np.zeros((doc_count, len(self.vocabulary)))
        for i, tokens in enumerate(tokenized_corpus):
            term_count = Counter(tokens)
            for term, count in term_count.items():
                if term in self.vocabulary:
                    term_index = self.vocabulary[term]
                    tf = count / len(tokens)
                    idf = log(doc_count / (1 + doc_freq[term]))
                    tfidf_matrix[i][term_index] = tf * idf

        return tfidf_matrix

    def transform(self, corpus):
        """Transform new corpus to term-document matrix."""
        doc_count = len(corpus)
        tfidf_matrix = np.zeros((doc_count, len(self.vocabulary)))

        for i, doc in enumerate(corpus):
            tokens = doc.lower().split()
            term_count = Counter(tokens)
            for term, count in term_count.items():
                if term in self.vocabulary:
                    term_index = self.vocabulary[term]
                    tf = count / len(tokens)
                    idf = log(len(corpus) / (1 + 1))  # Use placeholder IDF for unseen docs
                    tfidf_matrix[i][term_index] = tf * idf

        return tfidf_matrix
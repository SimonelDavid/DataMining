import os
import glob
import time
import numpy as np
import pandas as pd
from vstoica.lexicon import Lexicon
from sklearn.metrics import precision_score
TRAIN_DIR = "../aclImdb/train"
TEST_DIR = "../aclImdb/test"
def load_data(directory):
    """Load text data and labels."""
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(directory, label_type)
        for file_path in glob.glob(os.path.join(dir_path, '*.txt')):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels


lexicon = Lexicon()

print("\nStep 1: Loading training data...")
start_time = time.time()
train_texts, train_labels = load_data(TRAIN_DIR)
X_train = pd.DataFrame(train_texts, columns=['text'])
X_label = pd.DataFrame(train_labels, columns=['label'])

X_train['label'] = X_label

# Apply the preprocessing
X_train['processed_text'] = X_train['text'].apply(lexicon.preprocess_text)

# Predict sentiment on the processed text
X_train['predicted_sentiment'] = X_train['processed_text'].apply(lexicon.get_sentiment)

# Calculate precision
precision = precision_score(X_train['label'], X_train['predicted_sentiment'])

print("Precision of Sentiment Analysis:", precision)


# Step 2: Load testing data
# print("\nStep 2: Loading testing data...")
# start_time = time.time()
# test_texts, test_labels = load_data(TEST_DIR)

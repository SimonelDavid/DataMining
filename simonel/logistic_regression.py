import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths to data
TRAIN_DIR = "../aclImdb/train"
TEST_DIR = "../aclImdb/test"

def load_data(directory):
    print(f"Loading data from {directory}...")
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(directory, label_type)
        print(f"  Reading {label_type} reviews...")
        for file_path in glob.glob(os.path.join(dir_path, '*.txt')):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    print(f"Loaded {len(texts)} reviews from {directory}.")
    return texts, labels

# Load training and test data
print("Step 1: Loading training data...")
train_texts, train_labels = load_data(TRAIN_DIR)

print("Step 2: Loading testing data...")
test_texts, test_labels = load_data(TEST_DIR)

# Vectorize text data using TF-IDF
print("Step 3: Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
print("Vectorization complete.")

# Train Logistic Regression model
print("Step 4: Training Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train, train_labels)
print("Model training complete.")

# Evaluate model
print("Step 5: Evaluating the model...")
predictions = model.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(test_labels, predictions))

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(test_labels, predictions)
print(conf_matrix)

# Detailed Results
print("\nDetailed Results:")
true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()

print(f"True Positives (correctly identified positive reviews): {true_positives}")
print(f"True Negatives (correctly identified negative reviews): {true_negatives}")
print(f"False Positives (negative reviews incorrectly classified as positive): {false_positives}")
print(f"False Negatives (positive reviews incorrectly classified as negative): {false_negatives}")

total_positive_reviews = true_positives + false_negatives
total_negative_reviews = true_negatives + false_positives

print(f"\nPositive Reviews: {total_positive_reviews} (Correctly Identified: {true_positives}, Misclassified: {false_negatives})")
print(f"Negative Reviews: {total_negative_reviews} (Correctly Identified: {true_negatives}, Misclassified: {false_positives})")
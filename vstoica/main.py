import os
import glob
import time
import numpy as np
import psutil
import pandas as pd
from vstoica.lexicon import Lexicon
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
TRAIN_DIR = "../aclImdb/train"
TEST_DIR = "../aclImdb/test"
cpu_usages = []
memory_usages = []
start_times = []
end_times = []
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
def monitor_resources(step_name):
    """Logs CPU and memory usage for a specific step."""
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory in MB
    cpu_usages.append(cpu_usage)
    memory_usages.append(memory_usage)
    print(f"[{step_name}] CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage:.2f} MB")

def log_time(start_time, step_name):
    """Logs time taken for a step and stores it for overall analysis."""
    end_time = time.time()
    start_times.append(start_time)
    end_times.append(end_time)
    print(f"[{step_name}] Time Taken: {end_time - start_time:.2f} seconds")

lexicon = Lexicon()

print("\nStep 1: Loading data...")
start_time = time.time()
train_texts, train_labels = load_data(TEST_DIR)
log_time(start_time, "Loading Training Data")
monitor_resources("Loading Training Data")

X_train = pd.DataFrame(train_texts, columns=['text'])
X_label = pd.DataFrame(train_labels, columns=['label'])

X_train['label'] = X_label

# Apply the preprocessing
print("\nStep 2: Preprocess data...")
start_time = time.time()
X_train['processed_text'] = X_train['text'].apply(lexicon.preprocess_text)
log_time(start_time, "Preprocess Data")
monitor_resources("Preprocess Data")

# Predict sentiment on the processed text
print("\nStep 2: Predict sentiment...")
start_time = time.time()
X_train['predicted_sentiment'] = X_train['processed_text'].apply(lexicon.get_sentiment)
log_time(start_time, "Predict sentiment")
monitor_resources("Predict sentiment")
# Calculate precision
precision = precision_score(X_train['label'], X_train['predicted_sentiment'])

print("Precision of Sentiment Analysis:", precision)

report = classification_report(X_train['label'], X_train['predicted_sentiment'], target_names=['Negative', 'Positive'])

print("Classification Report:\n", report)
# Step 2: Load testing data
# print("\nStep 2: Loading testing data...")
# start_time = time.time()
# test_texts, test_labels = load_data(TEST_DIR)

conf_matrix = confusion_matrix(X_train['label'], X_train['predicted_sentiment'])
print("\nConfusion Matrix:")
print(conf_matrix)

# Detailed Results
true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()

print("\nDetailed Results:")
print(f"True Positives (correctly identified positive reviews): {true_positives}")
print(f"True Negatives (correctly identified negative reviews): {true_negatives}")
print(f"False Positives (negative reviews incorrectly classified as positive): {false_positives}")
print(f"False Negatives (positive reviews incorrectly classified as negative): {false_negatives}")

total_positive_reviews = true_positives + false_negatives
total_negative_reviews = true_negatives + false_positives

print(f"\nPositive Reviews: {total_positive_reviews} (Correctly Identified: {true_positives}, Misclassified: {false_negatives})")
print(f"Negative Reviews: {total_negative_reviews} (Correctly Identified: {true_negatives}, Misclassified: {false_positives})")

# Log overall metrics
total_time = sum(end - start for start, end in zip(start_times, end_times))
avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
peak_memory_usage = max(memory_usages)
print("\n[Overall Metrics]")
print(f"  Total Time Taken: {total_time:.2f} seconds")
print(f"  Average CPU Usage: {avg_cpu_usage:.2f}%")
print(f"  Peak Memory Usage: {peak_memory_usage:.2f} MB")

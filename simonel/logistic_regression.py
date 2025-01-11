import os
import glob
import time
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths to data
TRAIN_DIR = "../aclImdb/train"
TEST_DIR = "../aclImdb/test"

# Initialize tracking for overall metrics
cpu_usages = []
memory_usages = []
start_times = []
end_times = []

def system_info():
    """Logs system information: CPU and memory capacity."""
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Total memory in GB
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max  # Maximum frequency in MHz
    print(f"System Information:")
    print(f"  Total Memory: {total_memory:.2f} GB")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  CPU Max Frequency: {cpu_freq:.2f} MHz")

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

def overall_metrics():
    """Calculates and logs overall metrics."""
    total_time = sum(end - start for start, end in zip(start_times, end_times))
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    peak_memory_usage = max(memory_usages)
    print("\n[Overall Metrics]")
    print(f"  Total Time Taken: {total_time:.2f} seconds")
    print(f"  Average CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"  Peak Memory Usage: {peak_memory_usage:.2f} MB")

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

# System information
system_info()

# Track overall start time
overall_start_time = time.time()

# Step 1: Load training data
print("\nStep 1: Loading training data...")
start_time = time.time()
train_texts, train_labels = load_data(TRAIN_DIR)
log_time(start_time, "Loading Training Data")
monitor_resources("Loading Training Data")

# Step 2: Load testing data
print("\nStep 2: Loading testing data...")
start_time = time.time()
test_texts, test_labels = load_data(TEST_DIR)
log_time(start_time, "Loading Testing Data")
monitor_resources("Loading Testing Data")

# Step 3: Vectorize text data
print("\nStep 3: Vectorizing text data...")
start_time = time.time()
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
log_time(start_time, "Vectorizing Text Data")
monitor_resources("Vectorizing Text Data")

# Step 4: Train Logistic Regression model
print("\nStep 4: Training Logistic Regression model...")
start_time = time.time()
model = LogisticRegression()
model.fit(X_train, train_labels)
log_time(start_time, "Training Logistic Regression Model")
monitor_resources("Training Logistic Regression Model")

# Step 5: Evaluate the model
print("\nStep 5: Evaluating the model...")
start_time = time.time()
predictions = model.predict(X_test)
log_time(start_time, "Evaluating the Model")
monitor_resources("Evaluating the Model")

# Evaluation results
accuracy = accuracy_score(test_labels, predictions)
print(f"\nAccuracy: {accuracy}")
print("Classification Report:")
print(classification_report(test_labels, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predictions)
print("\nConfusion Matrix:")
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

# Log overall metrics
overall_metrics()
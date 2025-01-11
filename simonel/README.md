# Logistic Regression for Sentiment Analysis on IMDb Dataset

## Dataset Overview

The IMDb dataset contains 50,000 movie reviews split equally into two subsets:
- **Training Set (25,000 reviews):**
  - 12,500 positive reviews
  - 12,500 negative reviews
- **Testing Set (25,000 reviews):**
  - 12,500 positive reviews
  - 12,500 negative reviews

Each review is stored as a plain text `.txt` file, organized into separate folders (`pos` for positive and `neg` for negative reviews). The goal of this project is to classify the sentiment (positive or negative) of a review using machine learning.

---

## System Specifications

- **Total Memory:** 32.00 GB
- **CPU Cores:** 12
- **CPU Max Frequency:** 3504.00 MHz

---

## Logistic Regression Algorithm

### How It Works

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification tasks. It models the probability of a review belonging to a specific class (positive or negative) using the logistic function:

\[
P(y = 1|X) = \frac{1}{1 + e^{-(wX + b)}}
\]

Where:
- \( X \): Feature vector (e.g., TF-IDF representation of the review text).
- \( w \): Learned weights for each feature.
- \( b \): Bias term.

The algorithm minimizes a loss function (log-loss) during training to find the optimal weights \( w \) and bias \( b \). After training, it predicts the probability of each class for a given review and assigns the class with the highest probability.

---

## Results

### Performance Metrics

- **Accuracy:** 88.26%

### Classification Report

| Class           | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Negative (0)     | 0.88      | 0.88   | 0.88     | 12,500  |
| Positive (1)     | 0.88      | 0.88   | 0.88     | 12,500  |
| **Overall**      |           |        | **0.88** | 25,000  |

### Confusion Matrix

|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| **Actual Negative** | 11,013              | 1,487              |
| **Actual Positive** | 1,449               | 11,051             |

### Detailed Results

- **True Positives (correctly identified positive reviews):** 11,051
- **True Negatives (correctly identified negative reviews):** 11,013
- **False Positives (negative reviews incorrectly classified as positive):** 1,487
- **False Negatives (positive reviews incorrectly classified as negative):** 1,449

---

## Resource Utilization

### Time Taken

| Step                          | Time Taken (seconds) |
|-------------------------------|-----------------------|
| Loading Training Data         | 3.99                 |
| Loading Testing Data          | 4.13                 |
| Vectorizing Text Data         | 4.24                 |
| Training Logistic Regression  | 0.34                 |
| Evaluating the Model          | 0.01                 |
| **Total Time Taken**          | **12.70**            |

### CPU and Memory Usage

| Metric              | Value              |
|----------------------|--------------------|
| **Average CPU Usage** | 21.64%            |
| **Peak Memory Usage** | 367.88 MB         |

---

## Conclusion

The Logistic Regression model achieved an accuracy of **88.26%** on the IMDb dataset. While it performs well for this binary classification task, the following improvements could enhance performance:
- Using more sophisticated text representations like word embeddings (e.g., Word2Vec, GloVe).
- Trying more advanced machine learning models, such as Support Vector Machines (SVM) or neural networks.
- Hyperparameter tuning for the Logistic Regression model.

This analysis also provided insights into resource utilization, making it easier to evaluate system efficiency during the experiment.
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

### Implementation Details

The **LogisticRegressionCustom** class implements a custom logistic regression model with the following features:

- **Penalized Loss**: Supports L2 regularization to prevent overfitting.
- **Gradient Scaling**: Includes gradient normalization to prevent vanishing or exploding gradients.
- **Class Weight Balancing**: Adjusts for any class imbalance using weights.
- **Custom Initialization**: Initializes weights with small random values.
- **Convergence Check**: Stops early if the cost change is smaller than a predefined tolerance (`tol`).

#### Logistic Function
The algorithm models the probability of a review belonging to a specific class (positive or negative) using the logistic function:
\[
P(y = 1|X) = \frac{1}{1 + e^{-(wX + b)}}
\]

Where:
- \( X \): Feature vector (e.g., TF-IDF representation of the review text).
- \( w \): Learned weights for each feature.
- \( b \): Bias term.

#### Loss Function
The model minimizes the binary cross-entropy loss, with an optional L2 penalty term:
\[
J(w, b) = -\frac{1}{n} \sum_{i=1}^n \left[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right] + \frac{\lambda}{2n} \sum_{j=1}^m w_j^2
\]

Where:
- \( \hat{y}_i \): Predicted probability for the \(i\)-th sample.
- \( \lambda \): Regularization strength (`C` in the implementation).

---

## Text Vectorization

A custom **TfidfVectorizerCustom** was implemented to convert text reviews into numerical feature vectors. It includes:
- Term frequency-inverse document frequency (TF-IDF) scoring.
- Vocabulary limited to the top 3,000 most frequent terms.

---

## Results

### Performance Metrics

- **Accuracy:** 77.63%

### Classification Report

| Class           | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Negative (0)     | 0.82      | 0.70   | 0.76     | 12,500  |
| Positive (1)     | 0.74      | 0.85   | 0.79     | 12,500  |
| **Overall**      |           |        | **0.78** | 25,000  |

### Confusion Matrix

|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| **Actual Negative** | 8,782               | 3,718              |
| **Actual Positive** | 1,873               | 10,627             |

### Detailed Results

- **True Positives (correctly identified positive reviews):** 10,627
- **True Negatives (correctly identified negative reviews):** 8,782
- **False Positives (negative reviews incorrectly classified as positive):** 3,718
- **False Negatives (positive reviews incorrectly classified as negative):** 1,873

---

## Resource Utilization

### Time Taken

| Step                          | Time Taken (seconds) |
|-------------------------------|-----------------------|
| Loading Training Data         | 4.19                 |
| Loading Testing Data          | 4.16                 |
| Vectorizing Text Data         | 3.79                 |
| Training Logistic Regression  | 273.47               |
| Evaluating the Model          | 0.17                 |
| **Total Time Taken**          | **285.78**           |

### CPU and Memory Usage

| Metric              | Value              |
|----------------------|--------------------|
| **Average CPU Usage** | 29.38%            |
| **Peak Memory Usage** | 1,359.86 MB       |

---

## Conclusion

The custom Logistic Regression model achieved an accuracy of **77.63%** on the IMDb dataset. Key findings include:
- High recall for the positive class (0.85), indicating strong performance in identifying positive reviews.
- Moderate performance for the negative class, with lower recall (0.70).
- Resource-efficient training despite the custom implementation.

### Possible Improvements:
1. Explore more sophisticated feature representations like **word embeddings** (e.g., Word2Vec, GloVe).
2. Experiment with advanced algorithms such as Support Vector Machines (SVM) or deep learning models.
3. Perform hyperparameter tuning for learning rate, regularization strength, and batch size.

This analysis highlights the feasibility of using a fully custom implementation for text classification while offering insights into its performance and resource utilization.
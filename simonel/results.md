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

## Algorithm: Logistic Regression

### How It Works

Logistic Regression is a supervised machine learning algorithm that is commonly used for binary classification tasks. It models the probability of a review belonging to a specific class (positive or negative) using the logistic function:

\[
P(y = 1|X) = \frac{1}{1 + e^{-(wX + b)}}
\]

Where:
- \( X \) is the feature vector (e.g., TF-IDF representation of the review text).
- \( w \) are the learned weights for each feature.
- \( b \) is the bias term.

The algorithm minimizes a loss function (log-loss) during training to find the optimal weights \( w \) and bias \( b \). After training, it predicts the probability of each class for a given review and assigns the class with the highest probability.

### Workflow
1. **Data Loading**:
   - Read `.txt` files from the `train` and `test` directories.
   - Assign labels: `1` for positive reviews and `0` for negative reviews.

2. **Text Preprocessing**:
   - Convert text to lowercase.
   - Tokenize the text.
   - Use **TF-IDF (Term Frequency-Inverse Document Frequency)** to vectorize the reviews, representing text as numerical features.

3. **Model Training**:
   - Train a Logistic Regression model on the training data (25,000 reviews).

4. **Model Evaluation**:
   - Evaluate the model on the testing data (25,000 reviews) using metrics like accuracy, precision, recall, and F1-score.
   - Generate a confusion matrix to show the classification results.

---

## Results

### Accuracy
The overall accuracy of the Logistic Regression model on the test set is: 86.26%

### Classification Report
The performance for each class is summarized below:

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

## Key Insights

1. **Accuracy**: The model performs well, achieving an accuracy of 88.26%.
2. **Class Breakdown**:
   - Positive Reviews:
     - Correctly Classified: 11,051
     - Misclassified: 1,449
   - Negative Reviews:
     - Correctly Classified: 11,013
     - Misclassified: 1,487
3. **Balanced Performance**: The model shows balanced precision and recall for both positive and negative reviews, indicating it does not favor one class over the other.

---

## Conclusion

Logistic Regression is a simple yet effective algorithm for sentiment analysis. Using TF-IDF to vectorize text data, it achieves a high accuracy of 88.26% on the IMDb dataset. However, there is room for improvement, especially in reducing false classifications (False Positives and False Negatives).

Potential improvements could include:
- Using more advanced vectorization techniques (e.g., Word Embeddings).
- Trying more sophisticated models like SVM or neural networks.
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

##  Lexicon-Based

### Implementation Details

The **Lexicon** class implements a Lexicon-Based approach with the following features:

- **preprocess_text**: performs the text preprocessing for us: tokenization, stop words removal, and lemmatization.
- **get_sentiment**: checks if the positive score is grater than the negative one.

#### Preprocessing
Preprocessing is crucial for reducing noise and standardizing the text data to ensure better performance of the sentiment analysis model. The **Lexicon** class uses the following tools for preprocessing:
- **`nltk.tokenize.word_tokenize`**: For tokenizing text into words.
- **`nltk.corpus.stopwords`**: To remove common stop words in English.
- **`nltk.stem.WordNetLemmatizer`**: For lemmatizing the tokens to their root forms.

#### Getting the prediction
The prediction process involves loading the data, preprocessing it, and applying the sentiment analysis using the **Lexicon** class. Here’s an outline of the steps:

1. **Data Loading**: The `load_data` function reads text data and labels from the specified directory (`pos` for positive and `neg` for negative reviews).
2. **Preprocessing**: Each review is processed using the `preprocess_text` method to clean and standardize the text.
3. **Sentiment Prediction**: The processed text is analyzed with the `get_sentiment` method to predict whether the sentiment is positive or negative.

## Results

### Performance Metrics

- **Precision:** 62.16%
- **Accuracy:** 67%

### Classification Report

| Class           | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Negative (0)     | 0.79      | 0.47   | 0.59     | 12,500  |
| Positive (1)     | 0.62      | 0.88   | 0.73     | 12,500  |
| **Overall**      |           |        |          | 25,000  |

### Confusion Matrix

|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| **Actual Negative** | 5,842              | 1,560              |
| **Actual Positive** | 6,658              | 10,940             |

### Detailed Results

- **True Positives (correctly identified positive reviews):** 10,940
- **True Negatives (correctly identified negative reviews):** 5,842
- **False Positives (negative reviews incorrectly classified as positive):** 6,658
- **False Negatives (positive reviews incorrectly classified as negative):** 1,560

---

## Resource Utilization

### Time Taken

| Step                 | Time Taken (seconds) |
|----------------------|----------------------|
| Loading Testing Data | 2.13                 |
| Preprocess Data      | 988.68               |
| Predict sentiment    | 25.77                |
| **Total Time Taken** | **1016.58**           |

### CPU and Memory Usage

| Metric              | Value     |
|----------------------|-----------|
| **Average CPU Usage** | 0.77%     |
| **Peak Memory Usage** | 340.37 MB |

---

## Conclusion
In this project, we implemented a lexicon-based approach to sentiment analysis using the IMDb dataset. The results indicate that while the approach shows reasonable accuracy for positive sentiment prediction, it struggles with accurately identifying negative reviews. This imbalance in prediction suggests that the model is more tuned to recognizing positive sentiment cues, possibly due to inherent biases in the lexicon or the nature of the dataset.

### Key Insights:
- The **Precision** for positive reviews is higher than for negative reviews, indicating that the model is better at predicting when a review is positive but struggles with identifying negative reviews.
- The **Recall** for positive reviews is significantly higher than for negative ones, which reflects the model's propensity to classify more reviews as positive.
- The **Confusion Matrix** shows a substantial number of false positives, indicating that many negative reviews are misclassified as positive.


### Possible Improvements:
1. **Enhanced Lexicon Development**: 
   - Expand the sentiment lexicon to include more context-specific words or phrases that might be common in movie reviews.

2. **Contextual Analysis**:
   - Incorporate rules or heuristics to better understand context, such as handling sarcasm or double negatives which can change the sentiment meaning of phrases.

3. **Hybrid Approaches**: 
   - Augment the lexicon-based model with lightweight machine learning components to improve sentiment detection without requiring large amounts of labeled data.

4. **Domain-Specific Adjustments**: 
   - Tailor the lexicon specifically to the domain of movie reviews by analyzing common expressions and phrases used in this context, which might differ from general sentiment expressions.


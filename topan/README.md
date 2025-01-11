# Summary of Sentiment Analysis Implementation

## Overview
This document outlines the steps and processes implemented to analyze the sentiment of movie reviews from the **`aclImdb`** dataset. The project involves using a linguistic rule-based approach to classify reviews as positive, negative, or neutral.

---

## 1. Dataset Preparation
The **`aclImdb`** dataset contains reviews divided into the following folders:
- **`test/pos`**: Positive test reviews
- **`test/neg`**: Negative test reviews
- **`train/pos`**: Positive training reviews
- **`train/neg`**: Negative training reviews

Two important files in the dataset were used:
- **`imdb.vocab`**: Contains a list of words used in the dataset.
- **`imdbEr.txt`**: Contains numerical sentiment scores for each word in `imdb.vocab`.

---

## 2. Generating Word Lists
We generated extended lists of positive and negative words using the `imdb.vocab` and `imdbEr.txt` files:

1. **Process**:
   - Words with high sentiment scores (e.g., `> 0.5`) were classified as positive.
   - Words with low sentiment scores (e.g., `< -0.5`) were classified as negative.

2. **Output**:
   - **`positive_words.txt`**: List of positive words.
   - **`negative_words.txt`**: List of negative words.

This step allowed us to create dynamic word lists directly from the dataset.

---

## 3. Sentiment Analysis
The sentiment analysis algorithm determines whether a review is positive, negative, or neutral.

1. **Logic**:
   - Count the occurrences of positive and negative words in a review.
   - Classify the review based on the counts:
     - **Positive**: More positive words than negative words.
     - **Negative**: More negative words than positive words.
     - **Neutral**: Equal counts of positive and negative words.

2. **Additional Feature**:
   - **Sentiment Score**: Calculate a numeric score based on word counts:
     
     \[
     \text{Score} = \text{Positive Count} - \text{Negative Count}
     \]

3. **Implementation**:
   - Reviews are read from `.txt` files.
   - Sentiments are determined and results are written to a file.

4. **Output**:
   - **`results.txt`**: Contains sentiment classifications for each review.

---

## 4. Visualization
We added a visualization step to better understand the sentiment distribution:
- Created a histogram to display the frequency of positive, negative, and neutral reviews.
- Used **Matplotlib** to generate and display/save the chart.

---

## 5. Folder Analysis
The project was extended to analyze multiple folders:
- **`test/pos`**, **`test/neg`**
- **`train/pos`**, **`train/neg`**

1. Results for each folder are saved to a separate file or appended to a common file.
2. Comparisons between folders (e.g., positive vs. negative) were made possible.

---

## 6. Key Files
### Scripts
- **`extend_vocab.py`**: Generates `positive_words.txt` and `negative_words.txt`.
- **`sentiment_analysis.py`**: Contains the sentiment analysis logic.
- **`main.py`**: Executes analysis on specific folders and generates results.

### Outputs
- **`positive_words.txt`**: List of positive words.
- **`negative_words.txt`**: List of negative words.
- **`results.txt`**: Sentiment analysis results for reviews.
- **`sentiment_distribution.png`**: Visualization of sentiment distribution.

---

## Future Improvements
1. **Advanced Sentiment Analysis**:
   - Add weights for words (e.g., "fantastic" > "good").
   - Detect negations (e.g., "not bad" → positive).
2. **Interactive Features**:
   - Create a user interface for running the analysis.
   - Allow users to input custom reviews for sentiment analysis.
3. **Performance Optimization**:
   - Parallelize processing for large datasets.
   - Cache word counts for frequently analyzed reviews.

---

## Conclusion
The implemented sentiment analysis algorithm successfully classifies movie reviews using a rule-based approach. The generated results and visualizations provide meaningful insights into the dataset. The project can be further enhanced with advanced natural language processing techniques and user-friendly interfaces.


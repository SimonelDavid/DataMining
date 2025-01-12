# Porter Stemming and Stopword Removal for Sentiment Analysis on IMDb Dataset

## Overview
This document outlines the steps and processes used to analyze the sentiment of movie reviews from the **`aclImdb`** dataset. This implementation leverages the strengths of the Porter Stemming algorithm in order to easily match words to lists of positive and negative stems to aid in the classification of reviews.

---

## Dataset
The **`aclImdb`** dataset contains reviews divided into the following folders:
- **`test/pos`**: Positive test reviews
- **`test/neg`**: Negative test reviews
- **`train/pos`**: Positive training reviews
- **`train/neg`**: Negative training reviews

In addition, the **`nltk stopwords`** dataset is used during the filtering process to remove irrelevant words such as prepositions and conjunctions.

## Porter Stemmer
The algorithm uses a custom implementation of the Porter Stemmer, present in the `stemmer.py` file. This class exposes a main method, `stem()`, which takes a word and returns its stem based on the Porter Stemming algorithm. 

## Generating word lists (Training)
The training step involves generating two text files, `pos.txt` and `neg.txt`, each containing a list of words present in the training data that has been classified as either positive or negative.

1. **Filtering**
  - Training reviews are read in memory and tokenized into words. Punctuation marks are replaced by spaces, and the text is turned to lowercase.
  - Words not considered relevant (numbers and HTML tags) are removed.
  - Common stopwords (as per the `nltk stopwords` dataset) are filtered out.
  - Remaining words are stemmed using the Porter Stemmer.
  - These stems are manually filtered and turned into two predefined lists of positive and negative stems gathered from a smaller sample size of the training data.
  - Finally, to further filter the irrelevant words, new stems are matched with the two aforementioned lists and exported.

Through this approach, some manual work is required. This could be improved further by using the `imdb.vocab` and `imdbEr.txt` files from the dataset, which contain the words used in the dataset, as well as numerical sentiment scores. Based on these, a threshold could be set to filter out the purely positive and purely negative words.

2. **Output**
  - **`pos.txt`**: List of positive words.
  - **`neg.txt`**: List of negative words.

This training step allows us to expand the word lists by using actual words present in the reviews and automatically matching them onto a sentiment value.

## (Sentiment Analysis) Testing
The sentiment analysis algorithm determines whether a review is positive or negative.

1. **Logic**
  - The word lists generated during the training step are read into memory. Test data is then read one file at a time.
  - Irrelevant words and stopwords are filtered out.
  - Remaining words are stemmed using the Porter Stemmer.
  - These stems are tested against the word lists, positive words and negative words are counted separately.
  - A sentiment score is calculated for the review, based on the word counts:
    \[
    \text{Score} = \text{Positive Count} - \text{Negative Count * 1.75}
    \]
  - The review is classified based on the score:
    - **Positive**: Higher positive score than negative score.
    - **Negative**: Higher negative score than positive score.

2. **Caveats**
  - Due to the nature of the reviews, human communication and linguistics, negative words are weighted higher in the sentiment score. A negative review can contain a large number of positive words while still having a negative message, so each negative word is given a score of 1.75 to compensate. This measure reduces the number of false positives while not affecting the actual number of true positives.
  - Swear words can be censored and appear as a number of "*" characters. These are automatically considered negative words.
  - A stem is matched to the word lists by checking if any word from the lists starts with the stem. This helps catch more verbose and colorful language, and helps evening out spelling differences for compound words.

3. **Output**
  - **`output.txt`**: Contains the file name, positive score, negative score, sentiment classification and correct classification for each review.

## Results

### Performance Metrics
- **Accuracy**: 67.18%

### Correctness

| Type             | Correct   | Incorrect |  Total | Percent |
|------------------|-----------|-----------|--------|---------|
| Negative         |  8,870    | 3,630     | 12,500 | 70.96%  |
| Positive         |  7,924    | 4,576     | 12,500 | 63.39%  |
| **Overall**      | 16,794    | 8,206     | 25,000 | 67.18%  |

## Resource Utilization

### Time Taken

| Step                          | Time Taken (seconds) |
|-------------------------------|----------------------|
| Parsing Training Data         |  29.23               |
| Creating Word Lists           |   1.07               |
| Loading Testing Data          |   0.10               |
| Processing Testing Data       | 227.78               |
| **Total Time Taken**          | **258.18**           |

### CPU and Memory Usage

| Metric              | Value              |
|----------------------|--------------------|
| **Average CPU Usage** | 4.9%            |
| **Peak Memory Usage** | 248.53 MB       |

## Conclusion and Further Improvements:

The algorithm achieved an accuracy of **67.18%** on the IMDb dataset.

We find that although the negative score weighting was chosen in such a way that it would not affect true positives, the algorithm obtained a higher accuracy in detecting negative reviews. It also proves to be very resource efficient and relatively fast, despite having to process and hold up to 25000 text files in memory at its peak.

One point of improvement relies in the use of the vocabulary and sentiment error files built into the dataset.

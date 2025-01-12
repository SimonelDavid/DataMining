# Plan:
# Read pos/neg word lists
# Read test data
# Remove common words thru a stopwords list
# Stem remaining words
# Compare stemmed words to these lists and give the review a pos/neg score
# Classify the review based on that score.
# Export results to output file

import os
import sys
import time
from stemmer import Stemmer
from utils import log_time
from utils import log_resources
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

test_path = "../aclImdb/test"
pos_path = test_path + "/pos"
neg_path = test_path + "/neg"

stop_words = set(stopwords.words("english"))
alphabet = "abcdefghijklmnopqrstuvwxyz*"
stem = Stemmer()

# parse cli args
limit = -1

if len(sys.argv) > 2:
  if ("-l" in sys.argv or "--limit" in sys.argv) and sys.argv.index("-l") + 1 < len(sys.argv):
    limit_index = sys.argv.index("-l") if "-l" in sys.argv else sys.argv.index("--limit")
    if sys.argv[limit_index + 1].isnumeric():
      limit = int(sys.argv[limit_index + 1])

def filter_words(data):
  tokens = [word for word in word_tokenize(data.lower())]
  filtered_tokens = [word for word in tokens if word not in stop_words and word[0] in alphabet]
  return filtered_tokens


times = [time.time()]

with open("pos.txt", "r") as f:
  pos_words = f.read().split(",")
  pos_words.pop()

with open("neg.txt", "r") as f:
  neg_words = f.read().split(",")
  neg_words.pop()

times = log_time(times, "Reading sentiment word lists")
log_resources("Reading sentiment word lists")

limit = -1
if len(sys.argv) > 2 and (sys.argv[1] == "-l" or sys.argv[1] == "--limit") and sys.argv[2].isnumeric():
  limit = int(sys.argv[2])

out = open("output.txt", "w")
pos_score, neg_score = 0, 0

def classify_file(file, correct):
    # read file
    file_name = os.path.basename(file)
    f = open(file, "r")
    data = f.readline()
    f.close()

    # process words
    words = filter_words(data)
    stemmed_words = [stem.stem(word) for word in words]

    # compute sentiment scores
    pos_score, neg_score = 0, 0

    for word in stemmed_words:
      if word[0] == '*':
        # curse word, mark as negative
        neg_score += 1.75
        continue
      for suggestion in pos_words:
        if suggestion.startswith(word):
          pos_score += 1
          break
      for suggestion in neg_words:
        if suggestion.startswith(word):
          neg_score += 1.75
          break

    # classify review
    out.write(f"{file_name}: {{ positive score: {pos_score}, negative score: {neg_score} }}, classification: ")
    if pos_score > neg_score:
      out.write("positive")
    else:
      out.write("negative")
    out.write(f", correct: {correct}\n")

    # return correctness
    if correct == "pos" and pos_score > neg_score:
      return True
    elif correct == "neg" and neg_score > pos_score:
      return True
    return False

pos_correct, pos_total = 0, 0
neg_correct, neg_total = 0, 0

times = log_time(times, "Processing positive reviews")
log_resources("Processing positive reviews")

ct = 0
with os.scandir(pos_path) as pos_files:
  for file in pos_files:
    pos_total += 1
    ct += 1
    if limit > -1 and ct > limit:
      break
    if classify_file(file, "pos"):
      pos_correct += 1

times = log_time(times, "Processing negative reviews")
log_resources("Processing negative reviews")
ct = 0
with os.scandir(neg_path) as neg_files:
  for file in neg_files:
    neg_total += 1
    ct += 1
    if limit > -1 and ct > limit:
      break
    if classify_file(file, "neg"):
      neg_correct += 1

correct = pos_correct + neg_correct
total = pos_total + neg_total

times = log_time(times, "Finishing up")
log_resources("Finishing up")

out.write(f"Positive Correctness: {pos_correct}/{pos_total} ({pos_correct/pos_total*100:.2f}%)\n")
out.write(f"Negative Correctness: {neg_correct}/{neg_total} ({neg_correct/neg_total*100:.2f}%)\n")
out.write(f"Overall Correctness: {correct}/{total} ({correct/total*100:.2f}%)")
out.close()
print("Finished processing reviews.")
print(f"Positive Correctness: {pos_correct}/{pos_total} ({pos_correct/pos_total*100:.2f}%)")
print(f"Negative Correctness: {neg_correct}/{neg_total} ({neg_correct/neg_total*100:.2f}%)")
print(f"Overall Correctness: {correct}/{total} ({correct/total*100:.2f}%)")
print("Results exported to output.txt")
print(f"Time taken: {time.time() - times[0]:.2f} seconds")

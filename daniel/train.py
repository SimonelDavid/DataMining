# Plan:
# Read training data
# Remove common words thru a stopwords list
# Stem remaining words
# Create pos/neg word lists
# Export to input files

import os
import sys
import time
from stemmer import Stemmer
from utils import log_time
from utils import log_resources
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

train_path = "../aclImdb/train"
pos_path = train_path + "/pos"
neg_path = train_path + "/neg"

stop_words = set(stopwords.words("english"))
alphabet = "abcdefghijklmnopqrstuvwxyz"
punctuation = ".,!?;:()[]{}/\\'\""

words_in_pos_reviews = []
words_in_neg_reviews = []
translator = str.maketrans(punctuation, ' ' * len(punctuation))
stem = Stemmer()

# parse cli args
skip_suggestions = False
limit = -1

if len(sys.argv) > 1:
  if "-s" in sys.argv or "--skip-suggested" in sys.argv:
    skip_suggestions = True
  if ("-l" in sys.argv or "--limit" in sys.argv) and sys.argv.index("-l") + 1 < len(sys.argv):
    limit_index = sys.argv.index("-l") if "-l" in sys.argv else sys.argv.index("--limit")
    if sys.argv[limit_index + 1].isnumeric():
      limit = int(sys.argv[limit_index + 1])

def filter_words(data):
  tokens = [word for word in word_tokenize(data.lower().translate(translator))]
  filtered_tokens = [word for word in tokens if word not in stop_words and word[0] in alphabet]
  return filtered_tokens


times = [time.time()]
times = log_time(times, "Parsing words in positive reviews")
log_resources("Parsing words in positive reviews")

ct = 0
# parse words in positive reviews
with os.scandir(pos_path) as pos_files:
  for file in pos_files:
    ct += 1
    if limit != -1 and ct > limit:
      break
    f = open(file, "r")
    data = f.readline()
    f.close()
    
    words = filter_words(data)
    
    words_in_pos_reviews += [stem.stem(word) for word in words]

print("Finished parsing positive reviews.")
times = log_time(times, "Parsing words in negative reviews")
log_resources("Parsing words in negative reviews")

# parse words in negative reviews
with os.scandir(neg_path) as neg_files:
  for file in neg_files:
    ct += 1
    if limit != -1 and ct > limit:
      break
    f = open(file, "r")
    data = f.readline()
    f.close()
    
    words = filter_words(data)
    words_in_neg_reviews += words

print("Finished parsing negative reviews.")
times = log_time(times, "Creating word sentiment dictionary")
log_resources("Creating word sentiment dictionary")

pos_freq = Counter(words_in_pos_reviews)
neg_freq = Counter(words_in_neg_reviews)
word_sentiment = {}

for word in set(pos_freq.keys()).union(neg_freq.keys()):
  pos_score = pos_freq[word]
  neg_score = neg_freq[word]
  word_sentiment[word] = pos_score - neg_score

print("Finished creating word sentiment dictionary.")
times = log_time(times, "Exporting data")
log_resources("Exporting data")

pos_words = [word for word in word_sentiment.keys() if word_sentiment[word] > 0]
neg_words = [word for word in word_sentiment.keys() if word_sentiment[word] < 0]

# Extra filtering based on given words:
good = ["lik", "lov", "great", "amazing", "amaz", "cool", "satisf", "awesom", "strong", "sweet", "good", "fine", "grow", "interest", "surpris", "passion", "geniu", "humor", "fun", "entertain", "impress", "favorit", "favourit", "paradis", "fan", "excit", "uncompromis", "genre-def", "heart", "notab", "gourmet", "thrive", "chees", "satisfactori", "intellectu", "hot", "expert", "notori", "hope", "positiv", "realist", "suspens", "tastefulli", "heartwar", "praiseworthi", "gracefulli", "perfect", "dramat", "profound", "charismat", "eas", "sexi", "wealthi", "legend", "curiou", "impecab", "brilliant", "fascinat", "underestim", "desir", "aspir", "influen", "flawle", "recommen", "potenti", "outstan", "cute", "beauti", "honest", "underr", "myster", "personalit", "familiar", "fortun", "well-mean", "undoubt", "famou", "courag", "qualit", "poetic", "intim", "satiric", "testam", "romant", "phantasmagor", "succes", "lucki", "enthrall", "ambitious", "fantastic", "artistic", "charm", "superb"]

bad = ["bull", "bad", "wors", "dislik", "hat", "hati", "ridiculous", "dreadful", "underwhelm", "underwrit", "unmemorable", "crap", "unnecessary", "flaw", "unlike", "unrat", "edgi", "pretentious", "condescend", "poopie", "bizarre", "entitle", "horribl", "awful", "idiot", "stupid", "elitist", "unusual", "unrecogniz", "unrealistic", "diluted", "ripoff", "corn", "laughabl", "monoton", "unorthodox", "unimaginativ", "terribl", "rejected", "disinterest", "regretabl", "butchered", "hollow", "baffl", "boring", "ruined", "unbelieveabl", "irrelevant", "expectations", "long-wind", "embarass", "irrational", "turn-off", "low", "incorrect", "disgust", "unlik", "fluff", "underdeveloped", "mediocre", "nonsens", "strange", "ugly", "unnecessar", "dumb", "horrendous", "far-fetched", "poorl", "wast", "unfun", "break", "utter", "pathetic", "unfortun", "inconsistent", "incompetent", "\*\*\*\*", "ruin", "hinder", "phon", "nothing", "over-hyp", "turd", "sap"]

if skip_suggestions:
  print("Received --skip-suggested, will ignore further filtering...")
  with open("./pos.txt", "w") as f:
    for word in pos_words:
      f.write(f"{word}\n")

  with open("./neg.txt", "w") as f:
    for word in neg_words:
      f.write(f"{word}\n")

else:
  print("Filtering by suggested positive/negative words...")
  filtered_pos = set()
  filtered_neg = set()
  
  for word in pos_words:
      for suggestion in good:
        if word.startswith(suggestion):
          filtered_pos.add(word)
  
  for word in neg_words:
      for suggestion in bad:
          if word.startswith(suggestion):
            filtered_neg.add(word)
  
  with open("./pos.txt", "w") as f:
    for word in filtered_pos:
      f.write(f"{word},")

  with open("./neg.txt", "w") as f:
    for word in filtered_neg:
      f.write(f"{word},")

times = log_time(times, "Finishing up")
log_resources("Finishing up")

print("Done!")
print(f"Time taken: {time.time() - times[0]:.2f} seconds")

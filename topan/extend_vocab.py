def generate_extended_vocab(vocab_file_path, scores_file_path, threshold=0.5):
    """
    Generate extended lists of positive and negative words from dataset files.
    """
    # Read the dataset files
    with open(vocab_file_path, "r", encoding="utf-8") as vocab_file:
        vocab_content = vocab_file.read().splitlines()
    with open(scores_file_path, "r", encoding="utf-8") as scores_file:
        scores_content = scores_file.read().splitlines()

    # Convert scores to float
    scores = [float(score) for score in scores_content]

    # Determine positive and negative words based on a threshold
    positive_vocab = [vocab_content[i] for i, score in enumerate(scores) if score > threshold]
    negative_vocab = [vocab_content[i] for i, score in enumerate(scores) if score < -threshold]

    return positive_vocab, negative_vocab


if __name__ == "__main__":
    # Paths to files inside the dataset
    vocab_file_path = "../aclImdb/imdb.vocab"  # Update to your extracted path
    scores_file_path = "../aclImdb/imdbEr.txt"

    # Generate vocab lists
    positive_vocab, negative_vocab = generate_extended_vocab(vocab_file_path, scores_file_path)

    # Save results
    with open("../topan/positive_words.txt", "w", encoding="utf-8") as pos_file:
        pos_file.write("\n".join(positive_vocab))
    with open("../topan/negative_words.txt", "w", encoding="utf-8") as neg_file:
        neg_file.write("\n".join(negative_vocab))

    print(f"Generated positive and negative word lists. Saved in the 'topan' folder.")

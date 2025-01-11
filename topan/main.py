import os
from sentiment_analysis import basic_sentiment_analysis, positive_words, negative_words
import matplotlib.pyplot as plt


def analyze_reviews_in_folder(folder_path, output_file):
    """
    Analyze all .txt reviews in the specified folder and save results to an output file.
    """
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    review = file.read()
                    sentiment = basic_sentiment_analysis(review, positive_words, negative_words)
                    results.append((file_name, sentiment))
            except UnicodeDecodeError:
                print(f"Skipping file due to encoding issue: {file_name}")

    # Write results to the output file
    with open(output_file, "w", encoding="utf-8") as result_file:
        for file_name, sentiment in results:
            result_file.write(f"{file_name}: {sentiment}\n")

    print(f"Analysis complete. Results saved to {output_file}")


def plot_sentiment_distribution(results_file):
    """
    Plot a histogram for sentiment distribution and save it as an image.
    """
    with open(results_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    sentiments = [line.strip().split(": ")[1] for line in lines]

    # Count occurrences of each sentiment
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for sentiment in sentiments:
        sentiment_counts[sentiment] += 1

    # Plot histogram
    plt.bar(sentiment_counts.keys(), sentiment_counts.values())
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiments")
    plt.ylabel("Frequency")

    # Save the plot instead of showing it
    plt.savefig("../topan/grafice/sentiment_distribution.png")
    print("Plot saved as '../topan/grafice/sentiment_distribution.png'")


if __name__ == "__main__":
    # folder_to_analyze = "../aclImdb/test/neg"  # Update this path to your dataset folder
    # output_file = "../topan/results.txt"
    # analyze_reviews_in_folder(folder_to_analyze, output_file)
    plot_sentiment_distribution("../topan/results.txt")

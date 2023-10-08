from source.datacreator import Datacreator
from source.config import load_file_paths_configuration

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import numpy as np


# Script for data analysis to study the data and the machine learning model performance

file_paths_configuration = load_file_paths_configuration(
    "output/data/file_paths_config.json"
)

datacreator_with_duplicates = Datacreator(False)

datacreator_with_duplicates.openfile(file_paths_configuration.dialog_acts_path)
datacreator_with_duplicates.assign_class()
datacreator_with_duplicates.create_dataset()

datacreator_without_duplicates = Datacreator(True)

datacreator_without_duplicates.openfile(file_paths_configuration.dialog_acts_path)
datacreator_without_duplicates.assign_class()
datacreator_without_duplicates.create_dataset()

data = {
    "xtestwithduplicate": datacreator_with_duplicates.x_test,
    "xtrainwithduplicate": datacreator_with_duplicates.x_train,
    "ytestwithduplicate": datacreator_with_duplicates.y_test,
    "ytrainwithduplicate": datacreator_with_duplicates.y_train,
    "xtestwithoutduplicate": datacreator_without_duplicates.x_test,
    "xtrainwithoutduplicate": datacreator_without_duplicates.x_train,
    "ytestwithoutduplicate": datacreator_without_duplicates.y_test,
    "ytrainwithoutduplicate": datacreator_without_duplicates.y_train,
    "xwithduplicate": datacreator_with_duplicates.x,
    "ywithduplicate": datacreator_with_duplicates.y,
    "xwithoutduplicate": datacreator_without_duplicates.x,
    "ywithoutduplicate": datacreator_without_duplicates.y,
}

# 1 - Distribution of dialogs acts

# Get word count
word_counts_with_duplicates = Counter(data["ywithduplicate"])
word_counts_without_duplicates = Counter(data["ywithoutduplicate"])

# Convert word count into separate lists of words and frequencies
words_with_duplicates, frequencies_with_duplicates = zip(
    *word_counts_with_duplicates.items()
)
words_without_duplicates, frequencies_without_duplicates = zip(
    *word_counts_without_duplicates.items()
)

# Plt generate images of Distribution of Dialog Acts
plt.figure(figsize=(10, 6))
plt.bar(
    words_with_duplicates,
    frequencies_with_duplicates,
    alpha=0.5,
    color="g",
    label="Y withduplicate",
)
plt.bar(
    words_without_duplicates,
    frequencies_without_duplicates,
    alpha=0.5,
    color="b",
    label="Y withoutduplicate",
)
plt.gca().set(title="Frequency Histogram of DialogAct", ylabel="Frequency")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("output/images/distribution_dialog_act_comparison.jpg")

# Create the figure and axes for subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the histogram for Distribution of DialogActs for Ywithduplicate and Ywithoutduplicate
ax1.bar(words_with_duplicates, frequencies_with_duplicates, alpha=0.5, color="g")
ax1.set(title="Frequency Histogram of Y withduplicate", ylabel="Frequency")
ax1.tick_params(axis="x", rotation=90)
ax2.bar(words_without_duplicates, frequencies_without_duplicates, alpha=0.5, color="b")
ax2.set(title="Frequency Histogram of Y withoutduplicate", ylabel="Frequency")
ax2.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.savefig("output/images/distribution_dialogs_acts_separated.jpg")


plt.figure(figsize=(6, 6))
plt.pie(
    frequencies_with_duplicates,
    labels=words_with_duplicates,
    autopct="%1.1f%%",
    startangle=140,
)
plt.title("Distribution of Dialog Acts (with duplicates)")
plt.axis("equal")
plt.savefig("output/images/pie_chart_with_duplicates.jpg")

# 2 - Length of the utterances

# Calculate the average length of x for each dialogact in ywithduplicate

dialogsacts = set(data["ywithduplicate"])
dialogs_lengths = {}

# We get the lengths for each dialog act, then we calculate the average and add it to dialogs_lengths
for dialogact in dialogsacts:
    dialogsactsindexs = [
        i for i, y in enumerate(data["ywithduplicate"]) if y == dialogact
    ]
    dialogs_x_lengths = [len(data["xwithduplicate"][i]) for i in dialogsactsindexs]
    average_length = np.mean(dialogs_x_lengths)
    dialogs_lengths[dialogact] = average_length

# Also, we figure out the average length of every utterance (for every dialog act class)
x_lengthforall = [len(sentence) for sentence in data["xwithduplicate"]]
dialogs_lengths["average"] = np.mean(x_lengthforall)

# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(dialogs_lengths.keys(), dialogs_lengths.values())
plt.xlabel("Dialogs Acts")
plt.ylabel("Average Length Utterances")
plt.title("Average Length of X  WithDuplicates for DialogAct")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/images/length_utterances.jpg")

# 3 - Most used words

# We attach every utterance in text
text = " ".join(data["xwithduplicate"])

# Create object WorldCloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

# Calculate the top 10 words
word_counts = Counter(text.split())
top_10_words = word_counts.most_common(10)

plt.figure(figsize=(12, 6))

# Word cloud plus top 10 words
plt.subplot(1, 2, 1)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud most used words in utterances (xwithduplicates)")
plt.subplot(1, 2, 2)
words, frequencies = zip(*top_10_words)
plt.barh(words, frequencies, color="skyblue")
plt.gca().invert_yaxis()
plt.title("Top 10 Mentioned Words")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("output/images/most_words_used.jpg")

# 5 - Words per Dialog Act


dialogsacts = set(data["ywithduplicate"])

top_words_by_dialog_act = {}


def get_top_words(text, n):
    word_counts = Counter(text.split())
    top_words = dict(word_counts.most_common(n))
    return top_words


for dialogact in dialogsacts:
    filtered_phrases = [
        data["xwithduplicate"][i]
        for i, y in enumerate(data["ywithduplicate"])
        if y == dialogact
    ]
    text_for_dialogact = " ".join(filtered_phrases)
    number_of_words = 5
    top_words = get_top_words(text_for_dialogact, number_of_words)

    top_words_by_dialog_act[dialogact] = top_words

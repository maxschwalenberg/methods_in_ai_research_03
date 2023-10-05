from source.datacreator import Datacreator
from source.ml_model import LogisticRegressionModel

import matplotlib.pyplot as plt
from collections import Counter


# Script for data analysis to study the data and the machine learning model performance

filename = "data/dialog_acts.dat"
datacreator_with_duplicates = Datacreator(False)

datacreator_with_duplicates.openfile(filename)
datacreator_with_duplicates.assignClass()
datacreator_with_duplicates.createDataset()

datacreator_without_duplicates = Datacreator(True)

datacreator_without_duplicates.openfile(filename)
datacreator_without_duplicates.assignClass()
datacreator_without_duplicates.createDataset()

data = {"xtestwithduplicate": datacreator_with_duplicates.x_test, 
        "xtrainwithduplicate" : datacreator_with_duplicates.x_train,
        "ytestwithduplicate" : datacreator_with_duplicates.y_test,
        "ytrainwithduplicate" :datacreator_with_duplicates.y_train,
        "xtestwithoutduplicate": datacreator_without_duplicates.x_test, 
        "xtrainwithoutduplicate" : datacreator_without_duplicates.x_train,
        "ytestwithoutduplicate" : datacreator_without_duplicates.y_test,
        "ytrainwithoutduplicate" :datacreator_without_duplicates.y_train,
        }


# Get word count
word_counts_with_duplicates = Counter(data["ytestwithduplicate"])
word_counts_without_duplicates = Counter(data["ytestwithoutduplicate"])

# Convert word count into separate lists of words and frequencies
words_with_duplicates, frequencies_with_duplicates = zip(*word_counts_with_duplicates.items())
words_without_duplicates, frequencies_without_duplicates = zip(*word_counts_without_duplicates.items())

# Plt generate images
plt.figure(figsize=(10, 6))
plt.bar(words_with_duplicates, frequencies_with_duplicates, alpha=0.5, color='g', label='Ytest withduplicate')
plt.bar(words_without_duplicates, frequencies_without_duplicates, alpha=0.5, color='b', label='Ytest withoutduplicate')
plt.gca().set(title='Frequency Histogram of DialogAct', ylabel='Frequency')
plt.legend()
plt.xticks(rotation=90) 
plt.tight_layout()
plt.savefig("output/images/distributionDialogActComparison.jpg")

# Create the figure and axes for subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the histogram for Ytest withduplicate and Ytest without duplicate
ax1.bar(words_with_duplicates, frequencies_with_duplicates, alpha=0.5, color='g')
ax1.set(title='Frequency Histogram of Ytest withduplicate', ylabel='Frequency')
ax1.tick_params(axis='x', rotation=90)
ax2.bar(words_without_duplicates, frequencies_without_duplicates, alpha=0.5, color='b')
ax2.set(title='Frequency Histogram of Ytest withoutduplicate', ylabel='Frequency')
ax2.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig("output/images/distributionDialogsActsSeparated.jpg")





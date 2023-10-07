import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from source.config import load_file_paths_configuration
from collections import defaultdict


from source.datacreator import Datacreator

from source.ml_model import DecisionTreeModel, LogisticRegressionModel
from source.baseline import RuleBasedBaseline, MajorityClassBaseline

filenames_config = load_file_paths_configuration("output/data/file_paths_config.json")


evaluation_results = pd.DataFrame(
    columns=["model", "duplicates_removed", "accuracy", "precision", "recall"]
)


remove_duplicates_options = [False, True]
# loop through the different two datasets and evaluate all 4 models
for remove_duplicate in remove_duplicates_options:
    datacreator_instance = Datacreator(remove_duplicate)

    # load input data for both datacreator instances
    # and process the data to create the final dataset
    datacreator_instance.openfile(filenames_config.dialog_acts_path)
    datacreator_instance.assign_class()
    datacreator_instance.create_dataset()

    # MajorityClass Baseline
    majority_baseline = MajorityClassBaseline(datacreator_instance)
    majority_baseline.develop()

    # add results to dataframe
    evaluation_results.loc[len(evaluation_results)] = [
        "majority class baseline",
        remove_duplicate,
        majority_baseline.accuracy,
        majority_baseline.precision,
        majority_baseline.recall,
    ]

    # Rule-Based Baseline
    rule_based_baseline = RuleBasedBaseline(
        datacreator_instance, filenames_config.baseline_rules_path
    )
    rule_based_baseline.develop()

    # add results to dataframe
    evaluation_results.loc[len(evaluation_results)] = [
        "rule-based baseline",
        remove_duplicate,
        rule_based_baseline.accuracy,
        rule_based_baseline.precision,
        rule_based_baseline.recall,
    ]

    # ML Model 1
    decision_tree = DecisionTreeModel(datacreator_instance)
    decision_tree.develop()

    # add results to dataframe
    evaluation_results.loc[len(evaluation_results)] = [
        "decision_tree",
        remove_duplicate,
        decision_tree.accuracy,
        decision_tree.precision,
        decision_tree.recall,
    ]

    # ML Model 2
    logistic_regression = LogisticRegressionModel(datacreator_instance)
    logistic_regression.develop()

    # add results to dataframe
    evaluation_results.loc[len(evaluation_results)] = [
        "logistic_regression",
        remove_duplicate,
        logistic_regression.accuracy,
        logistic_regression.precision,
        logistic_regression.recall,
    ]


evaluation_results.to_csv(filenames_config.evaluation_results_path)
print(f"Save evaluation results to {filenames_config.evaluation_results_path}")


# Error Analysis

(
    incorrect_preds_logistic,
    x_testerrors_logistic,
    correct_ytest_logistic,
) = logistic_regression.get_errors()

(incorrect_preds_tree, x_testerrors_tree, correct_ytest_tree) = decision_tree.get_errors()

(
    incorrect_preds_baseline,
    x_testerrors_baseline,
    correct_ytest_baseline,
) = rule_based_baseline.get_errors()

# Unique labels (each dialog act) and we create a graphic for each model
dialogs_acts = np.unique(
    np.concatenate([correct_ytest_logistic, correct_ytest_tree, correct_ytest_baseline])
)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 1 - Logistic Regression
axs[0].hist(
    incorrect_preds_logistic,
    bins=len(dialogs_acts),
    alpha=0.5,
    color="red",
    label="Incorrect (Logistic)",
)
axs[0].set_xticks(range(len(dialogs_acts)))
axs[0].set_xticklabels(dialogs_acts, rotation=90)
axs[0].set_xlabel("Dialog Act")
axs[0].set_ylabel("Frequency")
axs[0].legend()
axs[0].set_title("Error in predictions (Logistic)")

# 2 - Decision Trees
axs[1].hist(
    incorrect_preds_tree,
    bins=len(dialogs_acts),
    alpha=0.5,
    color="blue",
    label="Incorrect (Tree)",
)
axs[1].set_xticks(range(len(dialogs_acts)))
axs[1].set_xticklabels(dialogs_acts, rotation=90)
axs[1].set_xlabel("Dialog Act")
axs[1].set_ylabel("Frequency")
axs[1].legend()
axs[1].set_title("Error in predictions (Tree)")

# 3 - Baseline Model
axs[2].hist(
    incorrect_preds_baseline,
    bins=len(dialogs_acts),
    alpha=0.5,
    color="green",
    label="Incorrect (Baseline)",
)
axs[2].set_xticks(range(len(dialogs_acts)))
axs[2].set_xticklabels(dialogs_acts, rotation=90)
axs[2].set_xlabel("Dialog Act")
axs[2].set_ylabel("Frequency")
axs[2].legend()
axs[2].set_title("Error in predictions (Baseline)")

# Obtain the max value
max_freq = max(
    max(axs[0].get_yticks()), max(axs[1].get_yticks()), max(axs[2].get_yticks())
)

# Use the same scale for each graph
for ax in axs:
    ax.set_ylim(0, max_freq)

plt.tight_layout()
plt.savefig("output/images/error_predictions_models.jpg")


# Function to create a word cloud from a list of errors for each model
def create_word_cloud_subplot(errors, title, position):
    wordcloud = WordCloud(width=200, height=200, background_color="white").generate(
        " ".join(errors)
    )
    plt.subplot(1, 3, position)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)


plt.figure(figsize=(15, 5))

create_word_cloud_subplot(x_testerrors_logistic, "Logistic Regression Errors", 1)
create_word_cloud_subplot(x_testerrors_tree, "Decision Tree Errors", 2)
create_word_cloud_subplot(x_testerrors_baseline, "Baseline Model Errors", 3)

plt.tight_layout()
plt.savefig("output/images/error_words_models.jpg")


# How the models perform with the words
def accuracy_per_word(x_test, y_test, preds):
    # Unique labels (each dialog act) and we create a graphic for each model
    stringxtestlog = " ".join(x_test).split()
    wordslog = np.unique(stringxtestlog)
    # For each word we save the corrects predictions [0] the failed predictions [1] and the percentage of accuracy [2]
    word_accuracy = {word: [0, 0, 0] for word in wordslog}

    for i in range(len(x_test)):
        x_instance = x_test[i]
        words = x_instance.split()
        if preds[i] == y_test[i]:  # if the prediction is good, we add 1 in 0
            for word in words:
                if word in word_accuracy:
                    word_accuracy[word][0] += 1

        else:  # if the prediction is good, we add 1 in 1
            for word in words:
                if word in word_accuracy:
                    word_accuracy[word][1] += 1

    # Get the accuracy of each word, by counting the good predictions and the bad predictions
    for word, counts in word_accuracy.items():
        total_appearances = counts[0] + counts[1]
        if total_appearances > 0:
            accuracy_percentage = (counts[0] / total_appearances) * 100
            word_accuracy[word][2] = accuracy_percentage

    return word_accuracy


def obtain_top_worse_words(word_accuracy, k=10, threshold=2):
    # Calculate the top k words with worse accuracy
    filtered_word_accuracy = [
        (word, accuracy)
        for word, accuracy in word_accuracy.items()
        if accuracy[1] > threshold
    ]
    sorted_word_accuracy = sorted(filtered_word_accuracy, key=lambda x: x[1][2])

    worst_k_words = sorted_word_accuracy[:k]
    return worst_k_words


def obtain_top_most_failed_words(word_accuracy, k=10):
    # Calculate the top k words with most mistakes
    sorted_word_accuracy = sorted(
        word_accuracy.items(), key=lambda x: x[1][1], reverse=True
    )
    top_k_words = sorted_word_accuracy[:k]
    return top_k_words


wordaccuracylog = accuracy_per_word(
    logistic_regression.x_test, logistic_regression.y_test, logistic_regression.preds
)
wordaccuracytree = accuracy_per_word(
    decision_tree.x_test, decision_tree.y_test, decision_tree.preds
)
wordaccuracybaseline = accuracy_per_word(
    rule_based_baseline.x_test, rule_based_baseline.y_test, rule_based_baseline.preds
)
# Each word accuracy has every word and it saves corrects [0] fails [1] and percentage [2]

# The topkwords worse percentage
worsewordslog = obtain_top_worse_words(wordaccuracylog)
worsewordstree = obtain_top_worse_words(wordaccuracytree)
worsewordsbaseline = obtain_top_worse_words(wordaccuracybaseline)

# The topkwords most failed
mostfailslog = obtain_top_most_failed_words(wordaccuracylog)
mostfailstree = obtain_top_most_failed_words(wordaccuracytree)
mostfailsbaseline = obtain_top_most_failed_words(wordaccuracybaseline)


max_length = max(len(worsewordslog), len(worsewordstree), len(worsewordsbaseline))

# File with empty lists
# worsewordslog += [('', [0, 0, 0])] * (max_length - len(worsewordslog))
# worsewordstree += [('', [0, 0, 0])] * (max_length - len(worsewordstree))
# worsewordsbaseline += [('', [0, 0, 0])] * (max_length - len(worsewordsbaseline))

min_length = min(len(worsewordslog), len(worsewordstree), len(worsewordsbaseline))

# Cut off until minimal distance
worsewordslog = worsewordslog[:min_length]
worsewordstree = worsewordstree[:min_length]
worsewordsbaseline = worsewordsbaseline[:min_length]

# Create Data Frame from 3 Models
data = {
    "Logistic Reg": [word[0] for word in worsewordslog],
    "Decision Trees": [word[0] for word in worsewordstree],
    "Baseline": [word[0] for word in worsewordsbaseline],
    "Accuracy (%) LR": [round(word[1][2], 2) for word in worsewordslog],
    "Accuracy (%) DT": [round(word[1][2], 2) for word in worsewordstree],
    "Accuracy (%) B": [round(word[1][2], 2) for word in worsewordsbaseline],
    "Corrects LR": [word[1][0] for word in worsewordslog],
    "Corrects DT": [word[1][0] for word in worsewordstree],
    "Corrects B": [word[1][0] for word in worsewordsbaseline],
    "Mistakes LR": [word[1][1] for word in worsewordslog],
    "Mistakes DT": [word[1][1] for word in worsewordstree],
    "Mistakes B": [word[1][1] for word in worsewordsbaseline],
}

df = pd.DataFrame(data)


# Figure MatPlotlib from the dataframe
fig, ax = plt.subplots(figsize=(8, 6))

# Create a Table from de DataFrame
table = pd.plotting.table(
    ax, df, loc="center", cellLoc="center", colWidths=[0.2] * len(df.columns)
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

ax.axis("off")
plt.savefig("output/images/worse_words_models.jpg", bbox_inches="tight", dpi=300)


def calculate_average_error_by_length(x_test, y_test, preds):
    # Create dictionaries to store predictions and labels by sentence length
    predictions_by_length = defaultdict(list)
    labels_by_length = defaultdict(list)

    # Fill the dictionaries with predictions and labels based on sentence length
    for i in range(len(x_test)):
        x_instance = x_test[i]
        length = len(x_instance.split())

        # Add predictions and labels to the dictionaries based on length
        predictions_by_length[length].append(preds[i])
        labels_by_length[length].append(y_test[i])

    # Calculate average accuracy for each sentence length
    accuracy_by_length = {}
    for length, preds in predictions_by_length.items():
        labels = labels_by_length[length]

        correct_count = sum(1 for pred, label in zip(preds, labels) if pred == label)
        total_count = len(preds)

        if total_count > 0:
            accuracy = correct_count / total_count
        else:
            accuracy = 0.0

        accuracy_by_length[length] = accuracy

    # Calculate the average error for each sentence length
    error_by_length = {
        length: 1.0 - accuracy for length, accuracy in accuracy_by_length.items()
    }
    sorted_error_by_length = dict(sorted(error_by_length.items()))

    return sorted_error_by_length


def plot_average_errors_by_length(error_by_length_list):
    # Figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Obtain minimums and maximums values
    all_lengths = []
    all_errors = []
    for error_by_length in error_by_length_list:
        lengths = list(error_by_length.keys())
        errors = list(error_by_length.values())
        all_lengths.extend(lengths)
        all_errors.extend(errors)

    # Using the same scale to notice the differences between models
    max_length = max(all_lengths)
    min_length = min(all_lengths)
    max_error = max(all_errors)
    min_error = min(all_errors)

    for i, error_by_length in enumerate(error_by_length_list):
        lengths = list(error_by_length.keys())
        errors = list(error_by_length.values())

        axs[i].scatter(lengths, errors, c="royalblue", marker="o", s=100)
        axs[i].set_xlabel("Sentence Length")
        axs[i].set_ylabel("Average Error")
        if i == 0:
            axs[i].set_title(f"Logistic Regression Plot")
        elif i == 1:
            axs[i].set_title(f"Decision Tree Plot")
        else:
            axs[i].set_title(f"Baseline Plot")
        axs[i].set_xticks(lengths)
        axs[i].set_xlim([min_length, max_length])
        axs[i].set_ylim([min_error, max_error])
        axs[i].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("output/images/utterance_relation.jpg", bbox_inches="tight", dpi=300)


error_by_lengthlog = calculate_average_error_by_length(
    logistic_regression.x_test, logistic_regression.y_test, logistic_regression.preds
)
error_by_lengthtree = calculate_average_error_by_length(
    decision_tree.x_test, decision_tree.y_test, decision_tree.preds
)
error_by_lengthbaseline = calculate_average_error_by_length(
    rule_based_baseline.x_test, rule_based_baseline.y_test, rule_based_baseline.preds
)

error_by_length_list = [error_by_lengthlog, error_by_lengthtree, error_by_lengthbaseline]
plot_average_errors_by_length(error_by_length_list)

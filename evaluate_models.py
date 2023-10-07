import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from source.config import load_file_paths_configuration


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
    rule_based_baseline = RuleBasedBaseline(datacreator_instance, filenames_config.baseline_rules_path)
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
incorrect_preds_tree, x_testerrors_tree, correct_ytest_tree = decision_tree.get_errors()
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

# 2 - Binary Trees
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
    max(axs[0].get_yticks()),
    max(axs[1].get_yticks()),
    max(axs[2].get_yticks())
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

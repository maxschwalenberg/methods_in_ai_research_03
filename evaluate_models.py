import pandas as pd

from source.datacreator import Datacreator

# from source.baseline import RuleBasedBaseline
from source.ml_model import DecisionTreeModel, LogisticRegressionModel


evaluation_results = pd.DataFrame(
    columns=["model", "duplicates_removed", "accuracy", "precision", "recall"]
)
save_results_in = "data/eval_results.csv"

data_filename = "data/dialog_acts.dat"


baseline_rules_file = "data/baseline_rules.json"

remove_duplicates_options = [False, True]
# loop through the different two datasets and evaluate all 4 models
for remove_duplicate in remove_duplicates_options:
    datacreator_instance = Datacreator(remove_duplicate)

    # load input data for both datacreator instances
    # and process the data to create the final dataset
    datacreator_instance.openfile(data_filename)
    datacreator_instance.assignClass()
    datacreator_instance.createDataset()

    # Baseline
    # baseline = Baseline(datacreator_instance)
    # baseline.evaluate()

    # Rule-Based Baseline
    """rule_based_baseline = RuleBasedBaseline(datacreator_instance)
    rule_based_baseline.loadRulesFile(baseline_rules_file)
    rule_based_baseline.evaluate()"""

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


evaluation_results.to_csv(save_results_in)
print(f"Save evaluation results to {save_results_in}")

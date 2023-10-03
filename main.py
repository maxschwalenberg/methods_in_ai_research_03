from source.datacreator import Datacreator
from source.baseline import RuleBasedBaseline, Baseline
from source.ml_model import DecisionTreeModel, LogisticRegressionModel
from source.dialog_management import DialogManagement
from source.config import setup

configuration = setup()

filename = "data/dialog_acts.dat"
datacreator_with_duplicates = Datacreator(False)
datacreator_without_duplicates = Datacreator(True)

# load input data for both datacreator instances
# and process the data to create the final dataset
datacreator_with_duplicates.openfile(filename)
datacreator_without_duplicates.openfile(filename)


datacreator_with_duplicates.assignClass()
datacreator_without_duplicates.assignClass()

datacreator_with_duplicates.createDataset()
datacreator_without_duplicates.createDataset()

baseline_rules_file = "data/baseline_rules.json"
# loop through the different two datasets and evaluate all 4 models
for datacreator_instance in [
    datacreator_with_duplicates,
    datacreator_without_duplicates,
]:
    # Baseline
    # baseline = Baseline(datacreator_instance)
    # baseline.evaluate()

    # Rule-Based Baseline
    rule_based_baseline = RuleBasedBaseline(datacreator_instance)
    rule_based_baseline.loadRulesFile(baseline_rules_file)
    rule_based_baseline.evaluate()

    # ML Model 1
    decision_tree = DecisionTreeModel(datacreator_instance)
    decision_tree.develop()
    decision_tree.show_results()

    # ML Model 2
    logistic_regression = LogisticRegressionModel(datacreator_instance)
    logistic_regression.develop()
    logistic_regression.show_results()
quit()
# show_results(logistic_regression)

# ex_sentence = "thank you"
# pred = logistic_regression.predict_single_sentence(ex_sentence)

# do example prediction
# print(f"sentence `{ex_sentence}` --> {pred}")


# create the dialog
dialog_system = DialogManagement(logistic_regression, configuration, debug=True)
dialog_system.run_dialog()

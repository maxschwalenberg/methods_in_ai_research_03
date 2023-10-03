from source.datacreator import Datacreator
from source.baseline import RuleBasedBaseline
from source.ml_model import DecisionTreeModel, LogisticRegressionModel
from source.dialog_management import DialogManagement
from source.config import setup

configuration = setup()

obj = Datacreator(False)
filename = "data/dialog_acts.dat"
obj.openfile(filename)

obj.assignClass()
obj.createDataset()

# Rule-Based Baseline
rule_based_baseline = RuleBasedBaseline(obj)
filename2 = "data/baseline_rules.json"
rule_based_baseline.loadRulesFile(filename2)
rule_based_baseline.evaluate()

"""# ML Model 1
decision_tree = develop(DecisionTreeModel(obj))
# show_results(decision_tree)"""

# ML Model 2
logistic_regression = LogisticRegressionModel(obj)
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

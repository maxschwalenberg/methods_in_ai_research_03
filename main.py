from source.datacreator import datacreator
from source.baseline import RuleBasedBaseline
from source.ml_model import DecisionTreeModel, LogisticRegressionModel
from source.developmentmodels import show_results, develop
from source.dialog_management import DialogManagement


obj = datacreator(False)
filename = "data/dialog_acts.dat"
obj.openfile(filename)

obj.assignClass()
obj.createDataset()

# Rule-Based Baseline
rule_based_baseline = RuleBasedBaseline(obj)
filename2 = "data/baseline_rules.json"
rule_based_baseline.loadRulesFile(filename2)
rule_based_baseline.evaluate()

# ML Model 1
decision_tree = develop(DecisionTreeModel(obj))
# show_results(decision_tree)

# ML Model 2
logistic_regression = develop(LogisticRegressionModel(obj))
# show_results(logistic_regression)

ex_sentence = "thank you"
pred = logistic_regression.predict_single_sentence(ex_sentence)

# do example prediction
print(f"sentence `{ex_sentence}` --> {pred}")


# create the dialog
dialog_system = DialogManagement(logistic_regression)
dialog_system.run_dialog()

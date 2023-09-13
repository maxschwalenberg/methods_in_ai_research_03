from source.datacreator import datacreator
from source.baseline import RuleBasedBaseline
from source.ml_model import DecisionTreeModel, LogisticRegressionModel


obj = datacreator(False)

obj.openfile("data/dialog_acts.dat")

obj.assignClass()
obj.createDataset()

# Rule-Based Baseline
rule_based_baseline = RuleBasedBaseline(obj)
rule_based_baseline.loadRulesFile("data/baseline_rules.json")
rule_based_baseline.test()

# ML Model 1
decision_tree = DecisionTreeModel(obj)
decision_tree.create_bag_of_words()
decision_tree.fit()
decision_tree.predict()
print(decision_tree.preds)

# ML Model 2
logistic_regression = LogisticRegressionModel(obj)
logistic_regression.create_bag_of_words()
logistic_regression.fit()

# print(obj.x_train.shape)

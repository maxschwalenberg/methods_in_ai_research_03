from source.datacreator import datacreator
from source.baseline import RuleBasedBaseline
from source.ml_model import DecisionTreeModel, LogisticRegressionModel
from source.results import show_results


obj = datacreator(False)
filename = 'C:/Users/Alex/Desktop/AI/Workspace/Methods of Research/methods_in_ai_research_03/data/dialog_acts.dat'
obj.openfile(filename)

obj.assignClass()
obj.createDataset()

# Rule-Based Baseline
rule_based_baseline = RuleBasedBaseline(obj)
filename2 = 'C:/Users/Alex/Desktop/AI/Workspace/Methods of Research/methods_in_ai_research_03/data/baseline_rules.json'
rule_based_baseline.loadRulesFile(filename2)
rule_based_baseline.test()

# ML Model 1
decision_tree = DecisionTreeModel(obj)
decision_tree.create_bag_of_words()
decision_tree.fit()
decision_tree.predict()
decision_tree.evaluate()
show_results(decision_tree)

# ML Model 2
logistic_regression = LogisticRegressionModel(obj)
logistic_regression.create_bag_of_words()
logistic_regression.fit()
logistic_regression.predict()
logistic_regression.evaluate()
show_results(logistic_regression)

from source.datacreator import datacreator
from source.baseline import RuleBasedBaseline
from source.ml_model import DecisionTreeModel, LogisticRegressionModel


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
print("Decision Tree:")
print("Accuracy: ", decision_tree.accuracy)
print("Precision: ", decision_tree.precision)
print("Recall: ", decision_tree.recall)
for label, metrics in decision_tree.class_metrics.items():
    print(f"{label}:")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")

# ML Model 2
logistic_regression = LogisticRegressionModel(obj)
logistic_regression.create_bag_of_words()
logistic_regression.fit()
logistic_regression.predict()
logistic_regression.evaluate()
print("Logistic Regression:")
print("Accuracy: ", logistic_regression.accuracy)
print("Precision: ", logistic_regression.precision)
print("Recall: ", logistic_regression.recall)
for label, metrics in logistic_regression.class_metrics.items():
    print(f"{label}:")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")

# print(obj.x_train.shape)
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
dt_accuracy, dt_precision, dt_recall, dt_class_metrics = decision_tree.evaluate()
print("Decision Tree Model:")
print(f"Accuracy: {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall: {dt_recall:.4f}")
print(f"Metrics per class:")
for label, metrics in dt_class_metrics.items():
    print(f"Class {label}:")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")

# ML Model 2
logistic_regression = LogisticRegressionModel(obj)
logistic_regression.create_bag_of_words()
logistic_regression.fit()
logistic_regression.predict()
lr_accuracy, lr_precision, lr_recall, lr_class_metrics = logistic_regression.evaluate()
print("Logistic Regression Model:")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"Metrics per class:")
for label, metrics in lr_class_metrics.items():
    print(f"{label}:")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")

# print(obj.x_train.shape)

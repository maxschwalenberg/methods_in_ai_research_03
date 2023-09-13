from code.datacreator import datacreator
from code.ml_model import DecisionTreeModel
from code.ml_model2 import LogisticRegressionModel


obj = datacreator(False)

obj.openfile("data/dialog_acts.dat")

obj.assignClass()
obj.createDataset()

#ML Model 1
decision_tree = DecisionTreeModel(obj)
decision_tree.create_bag_of_words()
decision_tree.fit()

#ML Model 2
logistic_regression = LogisticRegressionModel(obj)
logistic_regression.create_bag_of_words()
logistic_regression.fit()

print(obj.x_train.shape)

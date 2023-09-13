from code.datacreator import datacreator
from code.ml_model import DecisionTreeModel

obj = datacreator(False)

obj.openfile("data/dialog_acts.dat")

obj.assignClass()
obj.createDataset()

decision_tree = DecisionTreeModel(obj)
decision_tree.create_bag_of_words()
decision_tree.fit()

print(obj.x_train.shape)

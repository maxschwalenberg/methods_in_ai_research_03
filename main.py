from code.datacreator import datacreator

obj = datacreator(False)

obj.openfile("data/dialog_acts.dat")

obj.assignClass()
obj.createDataset()

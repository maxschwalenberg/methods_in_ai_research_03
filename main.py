from datacreator import datacreator

obj = datacreator(False)

obj.openfile("data/dialog_acts.dat")

obj.assignClass()

obj.preprocessData()

#print(obj.labeled_data)


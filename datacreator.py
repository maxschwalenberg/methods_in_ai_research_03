import numpy as np
from sklearn.model_selection import train_test_split

class datacreator:

    def __init__(self, remove_duplicate):
        self.remove_duplicate = remove_duplicate
        self.lines = []
        self.labeled_data = []
        

    def openfile (self, filename):
        file = open(filename, 'r')
        self.lines = file.readlines()
    
    
    def assignClass (self):
        for line in self.lines:
            label, text = (line.split(' ')[0], line.split(' ')[1:])
            text = " ".join(text)
            text = text.split('\n')[0]
            self.labeled_data.append([label, text])

    def preprocessData (self):
        x =np.array(np.array(self.labeled_data)[:, 1])
        y =np.array(np.array(self.labeled_data)[:, 0])
        print(x)




    


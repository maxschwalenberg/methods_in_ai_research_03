import numpy as np
from sklearn.model_selection import train_test_split


class Datacreator:
    def __init__(self, remove_duplicate):
        # function for removing duplicates
        self.remove_duplicate = remove_duplicate
        self.lines = []
        self.labeled_data = []
        self.dataset_created = False

    def openfile(self, filename):
        # defined in main.py, opens the training set
        file = open(filename, "r")
        self.lines = file.readlines()

    def assign_class(self):
        # check if you want to remove duplicates
        if self.remove_duplicate:
            self.lines = set(self.lines)
        # split the list into label and text
        for line in self.lines:
            # only split once - for label and corresponding sentence
            splitted_line = line.split(" ", maxsplit=1)
            label, text = (splitted_line[0], splitted_line[1])
            text = "".join(text)
            # remove newlines
            text = text.strip()

            self.labeled_data.append([label, text])

    def create_dataset(self):
        # make a numpy array

        self.x = np.array(self.labeled_data)[:, 1]
        self.y = np.array(self.labeled_data)[:, 0]
        # make the train and test data split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.15, random_state=1
        )
        self.dataset_created = True

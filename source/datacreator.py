import numpy as np
from sklearn.model_selection import train_test_split


class Datacreator:
    """The data preprocessor which preprocesses the data so it can be used for the models"""

    def __init__(self, remove_duplicate):
        """Initalizes the class and checks if the preprocessed data must contain duplicates

        Args:
            remove_duplicate (bool):
        """
        self.remove_duplicate = remove_duplicate
        self.lines = []
        self.labeled_data = []
        self.dataset_created = False

    def openfile(self, filename):
        """Opens the dataset so it can be loaded within the application

        Args:
            filename (str): The location of the dataset
        """
        file = open(filename, "r")
        self.lines = file.readlines()

    def assign_class(self):
        """Formats the dataset into labels and utterances"""
        # Check if you want to remove duplicates
        if self.remove_duplicate:
            self.lines = set(self.lines)
        # Split the list into label and text
        for line in self.lines:
            # Only split once - for label and corresponding sentence
            splitted_line = line.split(" ", maxsplit=1)
            label, text = (splitted_line[0], splitted_line[1])
            text = "".join(text)
            # Remove newlines
            text = text.strip()

            self.labeled_data.append([label, text])

    def create_dataset(self):
        """Formats the dataset so it can be used by the models"""

        # Make a numpy array and store the label and utterances separately
        self.x = np.array(self.labeled_data)[:, 1]
        self.y = np.array(self.labeled_data)[:, 0]
        # Split the dataset between training and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.15, random_state=1
        )

        # Let the main application know that the preprocessing has been successful
        self.dataset_created = True

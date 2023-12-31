from sklearn.metrics import accuracy_score, precision_score, recall_score
from source.datacreator import Datacreator
import numpy as np


# parent model class
class Model:
    def __init__(self, datacreator_instance: Datacreator) -> None:
        self.datacreator_instance = datacreator_instance
        self.fitted = False

        self.model: Model

    def predict(self, encoded_sentence):
        self.preds = self.model.predict(encoded_sentence)

    # creates the function so that the children can use it
    def predict_single_sentence(self, user_input: str):
        pass

    def evaluate(self):
        """
        Calculates different metrics for the model and saves the metrics as attributes of the class.
        """
        self.accuracy = accuracy_score(self.datacreator_instance.y_test, self.preds)
        self.precision = precision_score(
            self.datacreator_instance.y_test,
            self.preds,
            average="macro",
            zero_division=0.0,
        )
        self.recall = recall_score(
            self.datacreator_instance.y_test,
            self.preds,
            average="macro",
            zero_division=0.0,
        )

        precision_perclass = precision_score(
            self.datacreator_instance.y_test,
            self.preds,
            labels=np.unique(self.datacreator_instance.y_test),
            average=None,
            zero_division=0.0,
        )
        recall_perclass = recall_score(
            self.datacreator_instance.y_test,
            self.preds,
            labels=np.unique(self.datacreator_instance.y_test),
            average=None,
            zero_division=0.0,
        )

        self.class_metrics = {
            f"Class {label}": {
                "Precision": precision_perclass[i],
                "Recall": recall_perclass[i],
            }
            for i, label in enumerate(np.unique(self.datacreator_instance.y_test))
        }

    def show_results(self, information_per_class: bool = False):
        """
        Prints the evaluation results.

        Args:
            information_per_class (bool, optional): Set if metrics for each class should be printed. Defaults to False.
        """
        print(f"{self.model.__class__.__name__}:")
        print("Accuracy:", self.accuracy)
        print("Precision:", self.precision)
        print("Recall:", self.recall)

        # print additional information about each class if wanted
        if information_per_class:
            for label, metrics in self.class_metrics.items():
                print(f"{label}:")
                print(f"  Precision: {metrics['Precision']:.4f}")
                print(f"  Recall: {metrics['Recall']:.4f}")

    def get_errors(self):
        """Get information about the misclaficiations.

        Returns:
            tuple(list, list, list): returns information about the wrong predictions.
        """
        self.x_test = self.datacreator_instance.x_test
        self.y_test = self.datacreator_instance.y_test
        self.preds = self.preds

        incorrect_preds = []
        incorrect_x_test = []
        correct_y_test = []

        # We save the predictions where the model have made a mistake
        for i in range(len(self.y_test)):
            if self.preds[i] != self.y_test[i]:
                incorrect_preds.append(self.preds[i])
                incorrect_x_test.append(self.x_test[i])
                correct_y_test.append(self.y_test[i])

        return incorrect_preds, incorrect_x_test, correct_y_test

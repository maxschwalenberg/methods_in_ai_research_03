from sklearn.metrics import accuracy_score, precision_score, recall_score
from source.datacreator import datacreator
import numpy as np

#evaluates the models
class Model:
    def __init__(self, datacreator_instance: datacreator) -> None:
        self.datacreator_instance = datacreator_instance
        self.fitted = False

    def predict(self):
        self.preds = self.model.predict(self.x_test_encoded)
    
    #creates the function so that the children can use it 
    def predict_single_sentence(self, user_input: str):
        pass

    def evaluate(self):
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

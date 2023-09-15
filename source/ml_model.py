import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score


from source.datacreator import datacreator
from source.model import Model


class MLModel(Model):
    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)

    def create_bag_of_words(self):
        datacreator_instance = self.datacreator_instance
        if datacreator_instance.dataset_created:
            vectorizer = CountVectorizer()

            bag = vectorizer.fit_transform(
                np.concatenate(
                    (datacreator_instance.x_train, datacreator_instance.x_test)
                )
            )

            x_concatenated = bag.toarray()

            self.x_train_encoded = x_concatenated[
                : datacreator_instance.x_train.shape[0]
            ]
            self.x_test_encoded = x_concatenated[
                datacreator_instance.x_train.shape[0] :
            ]

        else:
            raise Exception("The dataset has not been created yet!")


class DecisionTreeModel(MLModel):
    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)

    def fit(self):
        model = tree.DecisionTreeClassifier()
        model.fit(self.x_train_encoded, self.datacreator_instance.y_train)

        self.model = model

    def predict(self):
        preds = self.model.predict(self.x_test_encoded)
        self.preds = preds
        return preds

    def evaluate(self):
        preds = self.predict()
        accuracy = accuracy_score(self.datacreator_instance.y_test, preds)
        precision = precision_score(
            self.datacreator_instance.y_test, 
            preds, 
            average = "macro",
            zero_division=0.0
        )
        recall = recall_score(
        self.datacreator_instance.y_test,
        preds,
        average="macro",
        zero_division=0.0
    )
        precision_perclass = precision_score(
            self.datacreator_instance.y_test, 
            preds, 
            labels=np.unique(self.datacreator_instance.y_test),
            average=None,
            zero_division=0.0
        )
        recall_perclass = recall_score(
            self.datacreator_instance.y_test,
            preds,
            labels=np.unique(self.datacreator_instance.y_test),
            average=None,
            zero_division=0.0
        )

        class_metrics = {
            f"{label}": {
                "Precision": precision_perclass[i],
                "Recall": recall_perclass[i],
            }
            for i, label in enumerate(np.unique(self.datacreator_instance.y_test))
        }

        return accuracy, precision, recall, class_metrics
    

class LogisticRegressionModel(MLModel):
    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)

    def fit(self):
        model = LogisticRegression(max_iter=1000000)
        model.fit(self.x_train_encoded, self.datacreator_instance.y_train)

        self.model = model

    def predict(self):
        preds = self.model.predict(self.x_test_encoded)

        self.preds = preds
        return preds
    
    def evaluate(self):
        preds = self.predict()
        accuracy = accuracy_score(self.datacreator_instance.y_test, preds)
        precision = precision_score(
            self.datacreator_instance.y_test, 
            preds, 
            average = "macro",
            zero_division=0.0
        )
        recall = recall_score(
        self.datacreator_instance.y_test,
        preds,
        average="macro",
        zero_division=0.0
    )
        precision_perclass = precision_score(
            self.datacreator_instance.y_test, 
            preds, 
            labels=np.unique(self.datacreator_instance.y_test),
            average=None,
            zero_division=0.0
        )
        recall_perclass = recall_score(
            self.datacreator_instance.y_test,
            preds,
            labels=np.unique(self.datacreator_instance.y_test),
            average=None,
            zero_division=0.0
        )

        class_metrics = {
            f"Class {label}": {
                "Precision": precision_perclass[i],
                "Recall": recall_perclass[i],
            }
            for i, label in enumerate(np.unique(self.datacreator_instance.y_test))
        }

        return accuracy, precision, recall, class_metrics
    

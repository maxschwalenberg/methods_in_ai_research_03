import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

from datacreator import datacreator
from model import Model


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

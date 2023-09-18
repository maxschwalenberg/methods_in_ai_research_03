import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree

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
            self.vectorizer = vectorizer

            x_concatenated = bag.toarray()

            self.x_train_encoded = x_concatenated[
                : datacreator_instance.x_train.shape[0]
            ]
            self.x_test_encoded = x_concatenated[
                datacreator_instance.x_train.shape[0] :
            ]

        else:
            raise Exception("The dataset has not been created yet!")

    def predict_single_sentence(self, user_input: str):
        # encode the normal string into bag of words
        encoded_user_input = self.vectorizer.transform([user_input])
        # predicted encoded sentence
        pred = self.model.predict(encoded_user_input)

        return pred[0]


class DecisionTreeModel(MLModel):
    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)

    def fit(self):
        model = tree.DecisionTreeClassifier()
        model.fit(self.x_train_encoded, self.datacreator_instance.y_train)

        self.model = model


class LogisticRegressionModel(MLModel):
    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)

    def fit(self):
        model = LogisticRegression(max_iter=1000000)
        model.fit(self.x_train_encoded, self.datacreator_instance.y_train)

        self.model = model

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree

from source.datacreator import datacreator
from source.model import Model


class MLModel(Model):
    #var :datacreator refers to the class datacreator
    def __init__(self, datacreator_instance: datacreator) -> None: #function returns none, so you cannot put in a wrong type
        #inherits from the model.py class Model
        super().__init__(datacreator_instance)


    def create_bag_of_words(self):
        #call upon this self instance to be able to change it
        datacreator_instance = self.datacreator_instance
        
        #CountVectorizer converts a collection of text documents to a matrix of token counts 
        if datacreator_instance.dataset_created:
            vectorizer = CountVectorizer()
            bag = vectorizer.fit_transform(
                np.concatenate(
                    (datacreator_instance.x_train, datacreator_instance.x_test)
                )
            )
            self.vectorizer = vectorizer
            
            x_concatenated = bag.toarray()
            #use the tokenized array to calculate how many row there are: the number of sentences 
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
        # predicted encoded sentence based on one of the model (Decisiontree/logisticregression)
        pred = self.model.predict(encoded_user_input)
        return pred[0]


class DecisionTreeModel(MLModel):
    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)
    
    #fits the model on the training data using a decision tree classifier
    def fit(self):
        model = tree.DecisionTreeClassifier()
        model.fit(self.x_train_encoded, self.datacreator_instance.y_train)
        self.model = model


class LogisticRegressionModel(MLModel):
    def __init__(self, datacreator_instance: datacreator) -> None:
        super().__init__(datacreator_instance)

    #fits the model on the training data using a logistic regression model 
    def fit(self):
        model = LogisticRegression(max_iter=1000000)
        model.fit(self.x_train_encoded, self.datacreator_instance.y_train)

        self.model = model

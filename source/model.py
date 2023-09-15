from sklearn.metrics import accuracy_score, precision_score, recall_score
from source.datacreator import datacreator



class Model:
    def __init__(self, datacreator_instance: datacreator) -> None:
        self.datacreator_instance = datacreator_instance
        self.fitted = False

    def predict(self):
        self.preds = self.model.predict(self.x_test_encoded)

    def evaluate(self):
        self.accuracy = accuracy_score(self.datacreator_instance.y_test, self.preds)
        self.precision = precision_score(self.datacreator_instance.y_test, self.preds, average="micro")
        self.recall = recall_score(self.datacreator_instance.y_test, self.preds, average='micro')
    
      

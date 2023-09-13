from code.datacreator import datacreator


class Model:
    def __init__(self, datacreator_instance: datacreator) -> None:
        self.datacreator_instance = datacreator_instance
        self.fitted = False

    def evaluate(self):
        pass

    def predict(self):
        pass

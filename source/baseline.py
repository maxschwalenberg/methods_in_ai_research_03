from source.datacreator import Datacreator
from source.model import Model
import json
import re


class RuleBasedBaseline(Model):
    """The rule-based baseline which predicts utterances based on given rules.
    """
    def __init__(self, datacreator_instance: Datacreator, rulesfile: str) -> None:
        """Preprocessing of the baseline

        Args:
            datacreator_instance (Datacreator): The datacreator containing the dataset
            rulesfile (str): The filelocation of the rules json file
        """
        super().__init__(datacreator_instance)
        self.rule_data = []
        self.load_rules_file(rulesfile)

    def load_rules_file(self, filename):
        """Loads the content of the rule file locally within the class object

        Args:
            filename (str): The filelocation of the rules json file
        """
        file = open(filename).read()
        self.rule_data = json.loads(file)

    def predict_single_utterance(self):
        """ Predicts a single utterance of the user via a prompt
        """
        print("Enter utterance: ")
        utterance = input()

        # Go trough each keyword and check if the given utterance
        # contains the keyword.
        for rule in self.rule_data["rules"]:
            for keyword in rule["utterances"]:
                if re.search(r"\b" + keyword + r"\b", utterance):
                    print(rule["intent"])
                    return
                
        # Print the null class if no keyword match took place
        print("null")

    def predict(self, input_list):
        """Predicts a whole list of utterances

        Args:
            input_list (list): A list of utterances
        """
        result_list = []

        # Predict every utterance in the list
        for utterance in input_list:
            has_found = False
            # Go trough each keyword and check if the given utterance
            # contains the keyword
            for rule in self.rule_data["rules"]:
                for keyword in rule["keywords"]:
                    if re.search(r"\b" + keyword + r"\b", utterance):
                        result_list.append(rule["intent"])
                        has_found = True
                        break
                # When a keyword match has been succesfull break out
                # the whole keyword loop to prevent multible classifications
                # of a single utterance
                if has_found:
                    break
            # If no keyword match has been found. Add the 
            # null class to the result list
            if not has_found:
                result_list.append("null")
        
        # Store the result list within the class object 
        # so it can later be used for evaluation
        self.preds = result_list

    def develop(self):
        """ The main process which is used to benchmark the baseline
        """
        self.predict(self.datacreator_instance.x_test)
        self.evaluate()


class MajorityClassBaseline(Model):
    """Classifies a given utterance with the most common class of the training dataset
    """
    def __init__(self, datacreator_instance: Datacreator) -> None:
        """Preprocesses the data and calculates the most common class

        Args:
            datacreator_instance (Datacreator): The datacreator which contains the dataset
        """
        super().__init__(datacreator_instance)
        self.majority = None
        self.find_majority(self.datacreator_instance.y_test)

    def find_majority(self, input_list):
        """Calculates the most common class of the given set

        Args:
            input_list (list): The dataset containing only labels/classes
        """

        # Create dictionary to count the frequency
        # of each class
        count_dict = {}
        # Go trough each class of the dataset
        for intent in input_list:
            # Add the class to the dataset with a 
            # counter when it does not exist
            if intent not in count_dict:
                count_dict[intent] = 1
            # Increment the counter when the 
            # class does exist within the dictionary
            else:
                count_dict[intent] += 1
        self.majority = max(count_dict, key=count_dict.get)


    def predict_single_utterance(self):
        """Predicts a single utterance of the user via a prompt
        """
        print("Enter utterance: ")

        # Because we predict the majority anyways we can disregard the input
        _ = input()
        print(self.majority)

    def predict(self, input_list):
        """Predicts a given list with the most common class of the dataset

        Args:
            input_list (list): The list containing the utterances
        """
        self.preds = [self.majority] * len(input_list)

    def develop(self):
        """ The main process which is used to benchmark the baseline
        """
        self.predict(self.datacreator_instance.x_test)
        self.evaluate()

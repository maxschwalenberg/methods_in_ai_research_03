from source.datacreator import Datacreator
from source.model import Model
import json

class RuleBasedBaseline(Model):
    def __init__(self, datacreator_instance: Datacreator) -> None:
        super().__init__(datacreator_instance)
        self.rule_data = []

    def loadRulesFile(self, filename):
        file = open(filename).read()
        self.rule_data = json.loads(file)

    def predict(self):
        print("Enter utterance: ")
        utterance = input()
        for rule in self.rule_data["rules"]:
            if utterance in rule["keywords"]:
                return rule["intent"]

    def evaluate(self, test_data=True):
        counter = 0.0
        correct_counter = 0.0
        input_set = []

        if test_data:
            input_set = zip(
                self.datacreator_instance.x_test, self.datacreator_instance.y_test
            )
        else:
            input_set = zip(
                self.datacreator_instance.x_train, self.datacreator_instance.y_train
            )

        for utterance, dialog_act in input_set:
            for rule in self.rule_data["rules"]:
                if (
                    any(keyword in utterance for keyword in rule["keywords"])
                    and dialog_act in rule["intent"]
                ):
                    correct_counter += 1
                    break
            counter += 1
        print(
            "Rule-based baseline accuracy: ",
            str(round(correct_counter / counter * 100, 2)) + "%",
        )

class MajorityClassCounter(Model):

    def __init__(self, datacreator_instance: Datacreator) -> None:
        super().__init__(datacreator_instance)

    def majorityCounter(self):
        count_dict = {}
        for x in self.datacreator_instance.y_train:
            if x not in count_dict:
                count_dict[x] = 1
            else:
                count_dict[x] += 1
        majority = max(count_dict, key=count_dict.get)
        print(majority)
        
            

        # ack = 0
        # affirm = 0
        # bye = 0
        # confirm = 0
        # deny = 0
        # hello = 0
        # inform = 0
        # negate = 0
        # null = 0
        # repeat = 0
        # reqalts = 0
        # reqmore = 0
        # request = 0
        # restart = 0
        # for x in datacreator.y_train:
        #     if x == "ack":
        #         ack += 1
        #     if x == "affirm":
        #         affirm += 1
        #     if x == "bye":
        #         bye += 1
        #     if x == "confirm":
        #         confirm += 1
        #     if x == "deny":
        #         deny += 1
        #     if x == "hello":
        #         hello += 1
        #     if x == "inform":
        #         inform += 1
        #     if x == "negate":
        #         negate += 1
        #     if x == "null":
        #         null += 1
        #     if x == "repeat":
        #         repeat += 1
        #     if x == "reqalts":
        #         reqalts += 1
        #     if x == "reqmore":
        #         reqmore += 1
        #     if x == "request":
        #         request += 1
        #     if x == "restart":
        #         restart += 1

        # dictionary = {
        #     "ack": ack,
        #     "affirm": affirm,
        #     "bye": bye,
        #     "confirm": confirm,
        #     "deny": deny,
        #     "hello": hello,
        #     "inform": inform,
        #     "negate": negate,
        #     "null": null,
        #     "reqalts": reqalts,
        #     "reqmore": reqmore,
        #     "request": request,
        #     "restart": restart,
        #     "repeat": repeat,
        # }
        # majority = max(dictionary, key=dictionary.get)

        # return majority

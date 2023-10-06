import pandas as pd
import json

from source.config import FilePathsConfig


class RestaurantLookup:
    def __init__(self, file_paths_config: FilePathsConfig) -> None:
        self.data = pd.read_csv(file_paths_config.extended_restaurant_info_path)
        self.additional_requirement_rules = json.load(
            open(file_paths_config.additional_requirement_rules_path)
        )

    def lookup(self, preferences: dict) -> pd.DataFrame:
        preferences_keys = list(preferences.keys())

        if "additional_requirement" in preferences_keys:
            preferences_keys.remove("additional_requirement")

        result_df = self.data
        for preference_key in preferences_keys:
            if preferences[preference_key] != "Any":
                result_df = result_df.loc[
                    result_df[preference_key] == preferences[preference_key]
                ]

        # apply inference
        if "additional_requirement" in list(preferences.keys()):
            additional_requirement = preferences["additional_requirement"]
            result_df = self.inference(result_df, additional_requirement)

        return result_df

    @staticmethod
    def match_rule(rules, row):
        is_match = False

        # values that are leading to a inference being made
        explanations = []

        for rule in rules:
            conditions_list = []

            for defined_conditions_key in rule:
                if defined_conditions_key == "consequence":
                    continue

                defined_condition_value = rule[defined_conditions_key]
                actual_value = row[defined_conditions_key]

                if defined_condition_value == actual_value:
                    conditions_list.append(True)
                else:
                    conditions_list.append(False)

            # check if all conditions are fulfilled
            all_conditions_fullfilled = all(conditions_list)
            if all_conditions_fullfilled:
                # check consequence
                consequence = rule["consequence"]

                if consequence:
                    is_match = True

                    # add values to explanations list
                    explanations += [
                        {k: v} for k, v in rule.items() if k != "consequence"
                    ]
                else:
                    # at this point the other rules dont need to be checked anymore because it is proven that the additional requirements doesnt hold
                    is_match = False
                    break

        if is_match:
            # also add the additional explanations if the consequence is false
            for rule in rules:
                if not rule["consequence"]:
                    explanations += [
                        {k: f"not {v}"} for k, v in rule.items() if k != "consequence"
                    ]

        return is_match, explanations

    def inference(
        self, results_df: pd.DataFrame, additional_requirement: str
    ) -> pd.DataFrame:
        # query rule defined in an external file as a dict
        rules = self.additional_requirement_rules[additional_requirement]
        matches = []

        for i in range(len(results_df)):
            row = results_df.iloc[i]
            # check if the additional requirement is fulfilled
            # for this, loop through all rules
            # if a "false consequence" is found - immediately continue

            is_match, _ = self.match_rule(rules, row)

            if is_match:
                matches.append(i)

        results_df = results_df.iloc[matches]
        return results_df

    def explain_inference(self, row: pd.DataFrame, additional_requirement: str):
        explanations: list[dict]

        _, explanations = self.match_rule(
            self.additional_requirement_rules[additional_requirement], row
        )

        # construct explanation string
        explanation_string = (
            f"The recommended restaurant is {additional_requirement} because"
        )
        for i, explanation in enumerate(explanations):
            if i == len(explanations) - 1 and len(explanations) != 1:
                explanation_string += " and"

            extraced_key = list(explanation.keys())[0]
            corresponding_value = list(explanation.values())[0]

            if extraced_key == "stay_length":
                explanation_string += (
                    f" the length of the stay is {corresponding_value}"
                )

            elif extraced_key == "crowdedness":
                explanation_string += f" the restaurant is {corresponding_value}"

            elif extraced_key == "food_quality":
                explanation_string += f" the food is {corresponding_value}"

            elif extraced_key == "pricerange":
                explanation_string += f" the served food is {corresponding_value}"

            elif extraced_key == "food":
                explanation_string += f" the served food is {corresponding_value}"

            if i == (len(explanations) - 1):
                explanation_string += "."
            else:
                # if we add the last explanation which means that we add an 'and',
                # we dont wanna insert a comma
                if not (i == len(explanations) - 2 and len(explanations) != 1):
                    explanation_string += ","

        return explanation_string

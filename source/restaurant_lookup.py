import pandas as pd
import json

from source.config import FilePathsConfig


class RestaurantLookup:
    """Defines functionalities to lookup restaurants and also inference based on given preferences."""

    def __init__(self, file_paths_config: FilePathsConfig) -> None:
        """Reads restaurant data and inference rules.

        Args:
            file_paths_config (FilePathsConfig): Contains used file paths.
        """
        self.data = pd.read_csv(file_paths_config.extended_restaurant_info_path)
        self.additional_requirement_rules = json.load(
            open(file_paths_config.additional_requirement_rules_path)
        )

    def lookup(self, preferences: dict) -> pd.DataFrame:
        """Finds fitting restaurants for the users preferences. Also performs inference of rules and the explanation of inferences.

        Args:
            preferences (dict): preferences dictionary

        Returns:
            pd.DataFrame: dataframe of restaurants that fit the requirements
        """
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
    def match_rule(
        rules: list[dict], row: pd.DataFrame
    ) -> tuple[bool, list[dict[str, str]]]:
        """checks if a given restaurant fulfills the given inference rules.

        Args:
            rules (list[dict]): defined rules of an additional requirement
            row (pd.DataFrame): single restaurant in a dataframe

        Returns:
            tuple[bool, list[dict[str, str]]]: returns if the restaurant matches the rules and a list of explanations if the restaurant matches
        """
        is_match = False

        # values that are leading to a inference being made
        explanations = []

        for rule in rules:
            conditions_list = []

            for condition in rule["conditions"]:
                defined_condition_value = condition["value"]
                actual_value = row[condition["category"]]

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
                        {condition["category"]: condition["value"]}
                        for condition in rule["conditions"]
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
                        {condition["category"]: f'not {condition["value"]}'}
                        for condition in rule["conditions"]
                    ]

        return is_match, explanations

    def inference(
        self, results_df: pd.DataFrame, additional_requirement: str
    ) -> pd.DataFrame:
        """Given a list of restaurant this method applies the inference of a specified requirement.

        Args:
            results_df (pd.DataFrame): dataframe of restaurants
            additional_requirement (str): additional preference, e.g. romantic

        Returns:
            pd.DataFrame: restaurant that match the rules
        """
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

    def check_for_contradiction(
        self, preferences: dict, additional_requirement: str
    ) -> tuple[bool, str]:
        """Check if a contradiction exists between the preferences and the additional requirement.

        Args:
            preferences (dict): dictionary of preferences
            additional_requirement (str): additional requirement like romantic

        Returns:
            tuple[bool, str]: clash_found and explanation why it clashes
        """
        explanations = []

        rules = self.additional_requirement_rules[additional_requirement]

        # initialize the condition clash as false
        clashing_condition = False
        for rule in rules:
            # check if a clash exists
            consequence = rule["consequence"]

            for condition in rule["conditions"]:
                category = condition["category"]
                value = condition["value"]

                # only check if preference exists in dictionary
                if category in preferences:
                    # only proceed check if preference is not 'Any'
                    if preferences[category] != "Any":
                        # if the condition holds then the consequence is true
                        if consequence:
                            if value != preferences[category]:
                                clashing_condition = True

                                explanations.append({category: preferences[category]})
                        # if the condition holds then the consequence is false
                        else:
                            if value == preferences[category]:
                                clashing_condition = True

                                explanations.append({category: preferences[category]})

        explanation_string = ""
        if clashing_condition:
            explanation_string = f"The additional requirement {additional_requirement} leads to a contradiction with the already given preferences because"
            explanation_string = self.construct_explanation_string(
                explanation_string, explanations, False
            )

        return clashing_condition, explanation_string

    def explain_inference(self, row: pd.DataFrame, additional_requirement: str) -> str:
        """Construct explanation why a restaurant fits the requirement.

        Args:
            row (pd.DataFrame): restaurant
            additional_requirement (str): additional requirement

        Returns:
            str: explanation why restaurant fulfills condition
        """
        explanations: list[dict]

        _, explanations = self.match_rule(
            self.additional_requirement_rules[additional_requirement], row
        )

        # construct explanation string
        explanation_string = (
            f"The recommended restaurant is {additional_requirement} because"
        )
        explanation_string = self.construct_explanation_string(
            explanation_string, explanations, True
        )
        return explanation_string

    @staticmethod
    def construct_explanation_string(
        explanation_string: str, explanations: list[dict], explain_inference: bool
    ) -> str:
        """Generates explanation why restaurant fits or clashes with additional requirement.

        Args:
            explanation_string (str): base explanation string
            explanations (list[dict]): basis for constructing the explanation string
            explain_inference (bool): if the inference or the clash needs to be explained

        Returns:
            str: explanation string
        """
        for i, explanation in enumerate(explanations):
            if i == len(explanations) - 1 and len(explanations) != 1:
                explanation_string += " and"

            extraced_key = list(explanation.keys())[0]
            corresponding_value = explanation[extraced_key]

            if extraced_key == "stay_length":
                if explain_inference:
                    explanation_string += (
                        f" the length of the stay is {corresponding_value}"
                    )
                else:
                    explanation_string += (
                        f" the expectedlength of the stay is {corresponding_value}"
                    )

            elif extraced_key == "crowdedness":
                if explain_inference:
                    explanation_string += f" the restaurant is {corresponding_value}"
                else:
                    explanation_string += (
                        f" the expected busyness is {corresponding_value}"
                    )

            elif extraced_key == "food_quality":
                if explain_inference:
                    explanation_string += f" the food is {corresponding_value}"
                else:
                    explanation_string += (
                        f" the expected food quality is {corresponding_value}"
                    )

            elif extraced_key == "pricerange":
                if explain_inference:
                    explanation_string += f" the restaurant is {corresponding_value}"
                else:
                    explanation_string += f" the expected price is {corresponding_value}"

            elif extraced_key == "food":
                if explain_inference:
                    explanation_string += f" the served food is {corresponding_value}"
                else:
                    explanation_string += (
                        f" the expected served food is {corresponding_value}"
                    )

            if i == (len(explanations) - 1):
                explanation_string += "."
            else:
                # if we add the last explanation which means that we add an 'and',
                # we dont wanna insert a comma
                if not (i == len(explanations) - 2 and len(explanations) != 1):
                    explanation_string += ","
        return explanation_string

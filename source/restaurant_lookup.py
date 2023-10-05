import pandas as pd
import random
import json


class RestaurantLookup:
    def __init__(self, restaurant_info_csv_path: str) -> None:
        self.data = pd.read_csv(restaurant_info_csv_path)
        self.additional_requirement_rules = json.load(
            open("output/data/additional_requirements.json")
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
        else:
            additional_requirement = ""

        result_df = self.inference(result_df, additional_requirement)

        return result_df

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

            is_match = False
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
                    else:
                        # at this point the other rules dont need to be checked anymore because it is proven that the additional requirements doesnt hold
                        is_match = False
                        break

            if is_match:
                matches.append(i)

        results_df = results_df.iloc[matches]
        return results_df

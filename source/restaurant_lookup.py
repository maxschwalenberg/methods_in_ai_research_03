import pandas as pd
import random


class RestaurantLookup:
    def __init__(self, restaurant_info_csv_path: str) -> None:
        self.data = pd.read_csv(restaurant_info_csv_path)

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
        rowsdrop = []
        for i in range(len(results_df)):
            row = results_df.iloc[i]
            price = row["pricerange"]
            area = row["area"]
            food = row["food"]
            quality = row["FoodQuality"]
            busyness = row["Crowdedness"]
            length_stay = row["StayLength"]

            if additional_requirement == "touristic":
                if food == "romanian":
                    rowsdrop.append(i)
                # cheap AND good -> negate -> not cheap or not good
                if price != "cheap" or quality != "good":
                    rowsdrop.append(i)

            elif additional_requirement == "assigned seats":
                if busyness != "busy":
                    rowsdrop.append(i)

            elif additional_requirement == "children":
                if length_stay == "long":
                    rowsdrop.append(i)

            elif additional_requirement == "romantic":
                if busyness == "busy":
                    rowsdrop.append(i)

                if length_stay != "long":
                    rowsdrop.append(i)

        # make rowsdrop only contain unique indices
        rowsdrop = list(set(rowsdrop))

        results_df = results_df.drop(results_df.index[rowsdrop])
        return results_df

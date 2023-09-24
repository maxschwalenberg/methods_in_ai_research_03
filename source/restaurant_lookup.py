import pandas as pd
import random



class RestaurantLookup:
    def __init__(self, restaurant_info_csv_path: str) -> None:
        self.data = pd.read_csv(restaurant_info_csv_path)

    def lookup(self, preferences: dict) -> pd.DataFrame:
        data_columns = self.data.columns.tolist()
        preferences_keys = list(preferences.keys())

        # assert that each given key of the preferences actually makes sense
        # --> is in the lookup tables categories
        for preference_key in preferences_keys:
            assert preference_key in data_columns

        result_df = self.data
        for preference_key in preferences_keys:
            if preferences[preference_key] != "Any":
                result_df = result_df.loc[
                result_df[preference_key] == preferences[preference_key]
            ]
        return result_df

import pandas as pd


class RestaurantLookup:
    def __init__(self, restaurant_info_csv_path: str) -> None:
        self.data = pd.read_csv(restaurant_info_csv_path)

        print(self.data)

    def transform_column_names(self):
        pass

    def lookup(self, preferences: dict):
        data_columns = self.data.columns.tolist()
        preferences_keys = list(preferences.keys())

        # assert that each given key of the preferences actually makes sense
        # --> is in the lookup tables categories
        for preference_key in preferences_keys:
            assert preference_key in data_columns

        # create the lookup query
        query = ""
        query += f"{preferences_keys[0]} == `{preferences[preferences_keys[0]]}`"

        for preference_key in preferences_keys[1:]:
            query += f" and {preference_key} == `{preferences[preference_key]}`"

        print(query)
        self.data["food"]
        self.data.query("Area == west")


db = RestaurantLookup("data/restaurant_info.csv")
db.lookup({"food": "spanish", "area": "west"})

import pandas as pd
import random


def add_new_properties(source_csv_path: str, target_csv_path: str):
    food_qualities = ["bad", "decent", "good"]
    crowdedness = ["busy", "not busy"]
    length_stay = ["short", "long", "normal"]

    data = pd.read_csv(source_csv_path)

    number_of_rows = len(data)

    new_food_column = []
    new_crowdedness_column = []
    new_stay_length_column = []

    for _ in range(number_of_rows):
        new_food_column.append(
            food_qualities[random.randint(0, len(food_qualities) - 1)]
        )
        new_crowdedness_column.append(
            crowdedness[random.randint(0, len(crowdedness) - 1)]
        )
        new_stay_length_column.append(
            length_stay[random.randint(0, len(length_stay) - 1)]
        )

    data["FoodQuality"] = new_food_column
    data["Crowdedness"] = new_crowdedness_column
    data["StayLength"] = new_stay_length_column

    data.to_csv(target_csv_path)


add_new_properties("data/restaurant_info.csv", "data/new_restaurant_info.csv")

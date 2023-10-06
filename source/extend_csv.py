import pandas as pd
import random

from source.config import load_file_paths_configuration


def add_new_properties(source_csv_path: str, target_csv_path: str):
    # function to extend the existing CSV-file by the required properties so the inference task can be done
    food_qualities = ["bad", "decent", "good"]
    crowdedness = ["busy", "not busy"]
    length_stay = ["short", "long", "normal"]

    data = pd.read_csv(source_csv_path)

    number_of_rows = len(data)

    new_food_column = []
    new_crowdedness_column = []
    new_stay_length_column = []

    for _ in range(number_of_rows):
        new_food_column.append(random.choice(food_qualities))
        new_crowdedness_column.append(random.choice(crowdedness))
        new_stay_length_column.append(random.choice(length_stay))

    data["food_quality"] = new_food_column
    data["crowdedness"] = new_crowdedness_column
    data["stay_length"] = new_stay_length_column

    data.to_csv(target_csv_path, index=False)


if __name__ == "__main__":
    file_paths_config = load_file_paths_configuration(
        "output/data/file_paths_config.json"
    )
    add_new_properties(
        file_paths_config.restaurant_info_path,
        file_paths_config.extended_restaurant_info_path,
    )

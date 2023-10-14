import pickle
from source.config import load_file_paths_configuration, load_configuration
from source.datacreator import Datacreator
from source.ml_model import DecisionTreeModel


def fit_and_save_model(path):
    filenames_config = load_file_paths_configuration("output/data/file_paths_config.json")

    datacreator_with_duplicates = Datacreator(False)

    # load input data for both datacreator instances
    # and process the data to create the final dataset
    datacreator_with_duplicates.openfile(filenames_config.dialog_acts_path)
    datacreator_with_duplicates.assign_class()
    datacreator_with_duplicates.create_dataset()

    # fit ML Model
    decision_tree = DecisionTreeModel(datacreator_with_duplicates)
    decision_tree.develop()

    pickle.dump(decision_tree, open(path, "wb"))


if __name__ == "__main__":
    fit_and_save_model("output/data/decision_tree.rf")

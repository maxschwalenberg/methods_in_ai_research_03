from source.dialog_management import DialogManagement
from source.config import load_configuration, load_file_paths_configuration
from source.datacreator import Datacreator
from source.ml_model import DecisionTreeModel


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
decision_tree.show_results()


configuration = load_configuration(filenames_config.dialog_config_path)


# create the dialog
dialog_system = DialogManagement(
    decision_tree, configuration, filenames_config, debug=False
)
dialog_system.run_dialog()

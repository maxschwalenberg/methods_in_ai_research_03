from source.dialog_management import DialogManagement
from source.config import setup
from source.datacreator import Datacreator
from source.ml_model import LogisticRegressionModel


filename = "data/dialog_acts.dat"
datacreator_with_duplicates = Datacreator(False)


# load input data for both datacreator instances
# and process the data to create the final dataset
datacreator_with_duplicates.openfile(filename)
datacreator_with_duplicates.assignClass()
datacreator_with_duplicates.createDataset()


# fit ML Model
logistic_regression = LogisticRegressionModel(datacreator_with_duplicates)
logistic_regression.develop()
logistic_regression.show_results()


configuration = setup()


# create the dialog
dialog_system = DialogManagement(logistic_regression, configuration, debug=True)
dialog_system.run_dialog()

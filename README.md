# 1. Installation
After cloning the repository, the requirements need to be installed. This is done using a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


# 2. Project Structure
## data/
The data repository contains the provided data from blackboard. This data is used for fitting the models and looking up restaurants.
It also contains the baseline_rules.json, which defines rules for the baseline model as well as the eval_results.csv, which is generated by running evaluate_models.py

## output/data
Contains some .json files, like the definition of inference rules and the configurations of the dialog manager and general file paths used for the project.
The evaluation_results.csv is also save in here and is generated by the evaluation script.

## output/images
Contains generated images of the data analysis and error analysis. Some are also used in the report. 


## source/

### config.py
Provides methods to load the dialog mangement config options and also the config where file paths are saved.

### model.py
Here, the 'Model' class is defined. It serves as the parent class for the baseline models and machine learning models.
TODO: make naming more clear with the .predict()

It also provides the evaluate() method to generate metrics like accuracy, precision and recall.

### baseline.py
Contains the baselines which are used to provide a benchmark with the machine learning models. They contain two types, namely the majority baseline and the rule-based baseline. The majority baseline classifies the given utterance always with the most common class within the dataset while the rule-based baseline uses the given rules in /data/baselinerules.json which will help classify the given utterance.

### ml_model.py
Defines the 'MLModel' class which implements the functionality to encode the training data to a bag of words representation to which both the machine learning models (decision tree & logistic regression) have access due to inheritance. It also provides the multiple other functionalities, the machine learning model classes only have to implement the respective .fit() method.
### datacreator.py
Contains the DataCreator class which is mainly responsible for the preprocessing process. It loads the dataset-file and converts it into a object class. Finally, the data is split between train and test data so it can be used for the classification models. 


### extend_csv.py
Extends the restaurant_info.csv provided on Blackboard by the required columns.


### restaurant_lookup.py
Defines the 'RestaurantLookup' class that is used to use the extracted preferences and to find matching restaurants for the recommendation.
It also implements the functionality to reason about the additional requirements of a restaurant like 'romantic' given a set of rules. In addition, a function to explain the inference is implemented.


### dialog_management.py
Implements the dialog manager. 

The 'DialogManagement' is responsible for keeping track of the current state and also saves important information, like the extracted preferences.

Furthermore, all possible states are implemented in this file, each having his own transition function.


# 3. Main Scripts
## evaluate_models.py -> implements 1a
to run: 
```bash
python evaluate_models.py
```

This script generates a .csv-file which contains the performance metrics for each of the models. Each model is evaluate with and without the removal of duplicate utterances. In addition, it is studied how is the performance of the models regarding some features (length utterance, dialog acts...).

The default output path of this file is data/eval_results.csv and is hardcoded in the script.


## data_analysis.py -> implements 1a
to run: 
```bash
python data_analysis.py
```

Data analysis focus on the dataset used in this dialog system. The code is used to study the distribution of the data and various aspects from the data preprocessing as the statistics.

## recommender.py -> implements 1a & 1b (dialog management)
to run: 
```bash
python recommender.py
```

This script fits a ML model on the data and uses the fitted model for the dialog management. Before the dialog is started, the user is asked to optionally change some configurations that influence the behavior/functionality of the dialog. 
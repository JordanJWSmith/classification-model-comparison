# FOML - Group 6 
To begin please install project dependecies by running in your shell:
    
```bash
pip install -r requirements.txt
```

We have provided a robust interface for running experiments on the models mentioned in this report. It is worth mentioning, running any model/command without any additional/optional parameters will run the final optimised model with a train and test set.
This README will first run through the application interface, followed by an explanation of the code structure.

# Commands Overview

Navigate to inside the "submission" directory from your command line. From here, the basic structure of the interface is as follows:

    python main.py [-h] [-a {accuracy,precision,f1,recall}] [-half]
               dataset_filename model

Positional arguments:

 - dataset_filename = CSV data file (**this will currently only accept 'wine_data.csv'**)
 - model = a single choice from the list {ALL, shared_covariance_model, knn, multiclass_logistic_regression, naive_bayes, multilayer_perceptron, random_forest, linearSVC}

Optional arguments:

 - -h = show the help screen, and explanation of interface commands
 - -a = choose an accuracy metric (default=precision, choices= {accuracy, precision, f1, recall})
 - -half = run with only half the training data

- --explore_dataset_histograms = display histograms and charts comparing data features (**will have no effect on model outputs - this is used only to show independent graphs that explore the data**)

**Plotting: Any plots created will be shown on screen after the code has finished executing. You will also be able to find these plots saved in the *submission/plots* directory**

## Sub-commands

Models are implemented as their own subcommands, each with their own parameters than can be entered. These subcommands can be viewed by triggering the help message (-h) after any subcommand.
For example:

    python main.py ALL -h

This will show the additional optional arguments that can be given to the 'ALL' sub-command.

Some generic commands may be used across all model sub-commands. These are the following:

- -h = show the help screen for a sub-command
- -k, --kfold = run the model using 10-fold cross validation (**may be used with all models except 'ALL'**)
- --per_class = returns results split by class (**may be used with all models except 'ALL'**)

Optional arguments and model specific arguments can be combined to give the user full control over the model and its parameters from the console/terminal. A more complex example of the interface's usage:

    python main.py -a recall -half wine_data.csv knn --per_class -k --n_neighbors 5

*This will run the knn model, selecting recall as the metric (-a) we would like to see, running the model with only half the training data (-half) over k-folds (-k) and with 5 neighbours ( --n_neighbors 5), and outputting the results split by class (--per_class).*

Sub-commands/Models that have additional option parameters are listed below. If a model is run with no optional parameters given, it will be run with its optimised paramters.

### Sub-command: ALL
***Runs all models with their optimised parameters on a train-test split***\
Output: *Plots models on the same chart against accuracy metric (-a) on the y-axis. All accuracy metrics per model printed into the console.*

- -s, --standardised = run with only standardised data
- -ns, --non-standardised = run with only non-standardised data
- --compare_standardisation = compare standardised to non-standardised

### Sub-command: knn
***Runs the knn model***\
Output: *Plots KNN model run with specified params against accuracy metric (-a) on the y-axis. All accuracy metrics printed into the console.*

- --n_neighbors N_NEIGHBORS = no. neighbours
- --test_params = compare values for n_neighbors (boolean flag, run 10-fold test across predefined set of inputs. Will ignore any sub-command/model specific parameters given)

### Sub-command: multiclass_logistic_regression
***Runs the multiclass logistic regression model***\
Output: *Plots MLR model run with specified params against accuracy metric (-a) on the y-axis. All accuracy metrics printed into the console.*

- -lr, --learning_rate LEARNING_RATE = specify learning rate
- -iter, --no_iterations NO_ITERATIONS = specify no. iterations
- --test_params = compare values for learning_rate and no_iterations (boolean flag, run 10-fold test across predefined set of inputs. Will ignore any sub-command/model specific parameters given)

### Sub-command: multilayer_perceptron
***Runs the multilayer perceptron model***\
Output: *Plots MLP model run with specified params against accuracy metric (-a) on the y-axis. All accuracy metrics printed into the console.*

- --hidden_layer_size HIDDEN_LAYER_SIZE = specify hidden layer size - ints separated by spaces
                        (default= 13 13 13)
- --activation {identity,logistic,tanh,relu} = select activation function - default="relu"
- --solver {lbfgs,sgd,adam} = select weight optimisation solver - default="lbfgs"
- --rand_state RAND_STATE = select random state - default=1
- --max_iter MAX_ITER = select max iterations - default=200
- --test_params = compare values for activation and solver models (boolean flag, run 10-fold test across predefined set of inputs. Will ignore any sub-command/model specific parameters given)

### Sub-command: random_forest
***Runs the random forest classifier model***\
Output: *Plots RFC model run with specified params against accuracy metric (-a) on the y-axis. All accuracy metrics printed into the console.*

- --no_estimators NO_ESTIMATORS = select no. estimators - default=100
- --max_features {auto,sqrt,log2} = the number of features to consider when looking for
                        the best split - default=auto
  
- --test_params = compare values for no_estimators (boolean flag, run 10-fold test across predefined set of inputs. Will ignore any sub-command/model specific parameters given)


### Sub-command: linearSVC
***Runs the linear support vector classifier model***\
Output: *Plots LinearSVC model run with specified params against accuracy metric (-a) on the y-axis. All accuracy metrics printed into the console.*

- --rand_state RAND_STATE = select random state - default=87
- --max_iter MAX_ITER = select max iterations - default=3000
- --C C = Regularisation strength, inversely proportional to C -
                        default=1.0
- --loss {squared_hinge,hinge} = loss function - default=squared_hinge
- --test_params = compare values for c_value (boolean flag, run 10-fold test across predefined set of inputs. Will ignore any sub-command/model specific parameters given)

## File structure

```bash
/submission
├── README.md
├── requirements.txt
├── __init__.py
├── calculate_result.py
├── fomlads.py
├── g6_helpers.py
├── get_data.py
├── knn.py
├── main.py
├── mlp.py
├── multiclass_logistic_regression.py
├── naive_bayes.py
├── plots
│   ├── plots saved here as .png
├── plotting.py
├── random_forest.py
├── run_models.py
├── shared_covariance_model.py
├── svm.py
└── wine_data.csv
```
Project files are in one directory for your convenience, with plots being saved within the 'plots' directory. Code is highly modular, and responsibility is separated in an organised & logical fashion.

### File roles
| File/directory                    | Responsibility                                                                                           |
|-----------------------------------|----------------------------------------------------------------------------------------------------------|
| README.md                         | This README file                                                                                         |
| requirements.txt                  | Project dependencies file                                                                                         |
| __ init __.py                     | Signify that the 'submission' directory may be imported as a python module                               |
| calculate_result.py               | Functions to process model results and return scores                                                     |
| fomlads.py                        | Functions modified directly from fomlads directory code                                                  |
| g6_helpers.py                     | Helper functions to handle train test splits, standardisation and performance metric calculations        |
| get_data.py                       | Functions that use helpers to return requested data                                                      |
| knn.py                            | KNN model code                                                                                           |
| main.py                           | Handles command line argument parsing and calls functions from run_models.py                             |
| mlp.py                            | Multilayer perceptron model code                                                                         |
| multiclass_logistic_regression.py | Multiclass logistic regresssion model code                                                               |
| naive_bayes.py                    | Naïve bayes model code                                                                                   |
| plots                             | Directory to store user generated plots                                                                  |
| plotting.py                       | Plotting functions                                                                                       |
| random_forest.py                  | Random forest model code                                                                                 |
| run_models.py                     | Handles logic to run chosen model from command line with chosen params/configuration then process result |
| shared_covariance_model.py        | Shared covariance model code                                                                             |
| svm.py                            | LinearSVC model code                                                                                     |
| wine_data.csv                     | Dataset file in csv format                                                                               |
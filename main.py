import sys
import os
import argparse
import matplotlib.pyplot as plt

from knn import knn
from shared_covariance_model import run_covariance_model
from naive_bayes import naive_bayes
from mlp import mlp
from calculate_result import get_metric_from_scores
from multiclass_logistic_regression import run_mlr_model
from random_forest import run_rfc
from svm import run_svm
from plotting import box_plot
import numpy as np
from run_models import run_all_models, run_model
from get_data import plot_histogram

from multiclass_logistic_regression import run_mlr_model

from plotting import box_plot, box_plot_comparison

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Set up argparse + arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset_filename', action='store', default='wine_data.csv', help='choose a csv file')
# parser.add_argument('model',
#                     choices=["ALL", "shared_covariance_model", "knn", "multiclass_logistic_regression", "naive_bayes",
#                              "multilayer_perceptron", "random_forest", "linearSVC"], type=str, action='store',
#                     help='enter a model to run or "ALL" to run all models')


parser.add_argument('-a', '--accuracy_metric', action='store', type=str,
                    choices=['accuracy', 'precision', 'f1', 'recall'],
                    default='precision', help='accuracy metric - default = precision')

parser.add_argument('-half', action='store_true',
                    default=False, help='run with only half the data points')

sub_parsers = parser.add_subparsers(help='enter model to run after specifying filename and your optional arguments',
                                    dest='model')

#DATA
parser.add_argument('--explore_dataset_histograms', action='store_true',
                    default=False, help='display histograms and charts comparing data features')

# ALL
parse_all = sub_parsers.add_parser('ALL', help='run all models with their optimal params')
std = parse_all.add_mutually_exclusive_group()
std.add_argument('-s', '--standardised', action='store_true', help='run with only standardised data')
std.add_argument('-ns', '--non-standardised', action='store_true', help='run with only non-standardised data')
std.add_argument('--compare_standardisation', default=False, action='store_true', help='compare standardised to '
                                                                                       'non-standardised')


# SCM
parse_scm = sub_parsers.add_parser('shared_covariance_model', help='run shared covariance model')
parse_scm.add_argument('-k', '--kfold', default=False, action='store_true', help='run with 10 fold cross validation')
parse_scm.add_argument('--per_class', default=False, action='store_true', help='return per class scores')

# KNN
parse_knn = sub_parsers.add_parser('knn', help='run knn model')
parse_knn.add_argument('--n_neighbors', type=int, help="no. neighbours")
parse_knn.add_argument('-k', '--kfold', default=False, action='store_true', help='run with 10 fold cross validation')
parse_knn.add_argument('--per_class', default=False, action='store_true', help='return per class scores')
parse_knn.add_argument('--test_params', default=False, action='store_true',
                       help='iterate through different hyperparameters')

# MLR
parse_mlr = sub_parsers.add_parser('multiclass_logistic_regression', help='run mlr model')
parse_mlr.add_argument('-k', '--kfold', default=False, action='store_true', help='run with 10 fold cross validation')
parse_mlr.add_argument('-lr', '--learning_rate', default=0.93, type=float,
                       action='store', help='specify learning rate')
parse_mlr.add_argument('-iter', '--no_iterations', default=14, type=int, action='store',
                       help='specify no. iterations')
parse_mlr.add_argument('--per_class', default=False, action='store_true', help='return per class scores')
parse_mlr.add_argument('--test_params', default=False, action='store_true',
                       help='iterate through different hyperparameters')

# NB
parse_nb = sub_parsers.add_parser('naive_bayes', help='run naive bayes model')
parse_nb.add_argument('-k', '--kfold', default=False, action='store_true', help='run with 10 fold cross validation')
parse_nb.add_argument('--per_class', default=False, action='store_true', help='return per class scores')

# MP
parse_mp = sub_parsers.add_parser('multilayer_perceptron', help='run multilayer perceptron model')
parse_mp.add_argument('-k', '--kfold', default=False, action='store_true', help='run with 10 fold cross validation')
parse_mp.add_argument('--hidden_layer_size', default=[13, 13, 13], type=int, action='store', nargs="+",
                      help='specify hidden layer size - ints separated by spaces (default= 13 13 13)')
parse_mp.add_argument('--activation', default='tanh', choices=['identity', 'logistic', 'tanh', 'relu'], action='store',
                      help='select activation function - default="tanh"')
parse_mp.add_argument('--solver', default='lbfgs', choices=['lbfgs', 'sgd', 'adam'], action='store',
                      help='select weight optimisation solver - default="lbfgs"')
parse_mp.add_argument('--rand_state', default=1, type=int, action='store',
                      help='select random state - default=1')
parse_mp.add_argument('--max_iter', default=200, type=int, action='store',
                      help='select max iterations - default=200')
parse_mp.add_argument('--per_class', default=False, action='store_true', help='return per class scores')
parse_mp.add_argument('--test_params', default=False, action='store_true',
                      help='iterate through different hyperparameters')

# RFC
parse_rfc = sub_parsers.add_parser('random_forest', help='run random forest model')
parse_rfc.add_argument('-k', '--kfold', default=False, action='store_true', help='run with 10 fold cross validation')
parse_rfc.add_argument('--no_estimators', default=100, type=int, action='store',
                       help='select no. estimators - default=100')
parse_rfc.add_argument('--max_features', default="auto", choices=['auto', 'sqrt', 'log2'], action='store',
                       help='the number of features to consider when looking for the best split - default=auto')
parse_rfc.add_argument('--per_class', default=False, action='store_true', help='return per class scores')
parse_rfc.add_argument('--test_params', default=False, action='store_true',
                       help='iterate through different hyperparameters')

# LSVC
parse_lsvc = sub_parsers.add_parser('linearSVC', help='run linearSVC (SVM) model')
parse_lsvc.add_argument('-k', '--kfold', default=False, action='store_true', help='run with 10 fold cross validation')
parse_lsvc.add_argument('--rand_state', default=87, type=int, action='store',
                        help='select random state - default=87')
parse_lsvc.add_argument('--max_iter', default=3000, type=int, action='store',
                        help='select max iterations - default=3000')
parse_lsvc.add_argument('--C', default=1.0, type=float, action='store',
                        help='Regularisation strength, inversely proportional to C - default=1.0')
parse_lsvc.add_argument('--loss', default="squared_hinge", choices=['squared_hinge', 'hinge'], action='store',
                        help='loss function - default=squared_hinge')
parse_lsvc.add_argument('--per_class', default=False, action='store_true', help='return per class scores')
parse_lsvc.add_argument('--test_params', default=False, action='store_true',
                        help='iterate through different hyperparameters')

args = parser.parse_args()

# print(vars(args))

args = vars(args)
print("Selected parameters: ", args)

if args['dataset_filename'] != 'wine_data.csv':
    print("wine_data.csv is the only currently accepted filename")
    sys.exit()

if args['explore_dataset_histograms']:
    plot_histogram()

if args['model'] == 'ALL':
    run_all_models(args['standardised'], args['non_standardised'], args['compare_standardisation'],
                   args['accuracy_metric'], args['half'])
else:
    run_model(args['model'], args)

# Show all plots
plt.show()

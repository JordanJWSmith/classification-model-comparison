from knn import knn, test_knn
from shared_covariance_model import run_covariance_model
from naive_bayes import naive_bayes
from mlp import mlp, test_mlp
from calculate_result import get_metric_from_scores
from multiclass_logistic_regression import run_mlr_model, optimsation_test
from random_forest import run_rfc, run_test_params_rfc
from svm import run_svm, test_svm
import numpy as np

from multiclass_logistic_regression import run_mlr_model

from plotting import box_plot, box_plot_comparison


# All models can only be run with optimal model params. user can test standardisation, and run with half data
# + specific accuracy metric to output

def run_all_models(standardise, non_standardise, compare_std, accuracy_metric, half_data):
    if compare_std:
        cvmdl_mean_nonstd, cvmdl_scores_nonstd = run_covariance_model(standardise=False, half=half_data)
        cvmdl_acc_nonstd = get_metric_from_scores(cvmdl_scores_nonstd, accuracy_metric)
        print("Shared Covariance Model (Non-Standardised)", cvmdl_mean_nonstd)

        knn_mean_nonstd, knn_scores_nonstd = knn(standardise=False, half=half_data)
        knn_acc_nonstd = get_metric_from_scores(knn_scores_nonstd, accuracy_metric)
        print("KNN (Non-Standardised)", knn_mean_nonstd)

        nb_mean_nonstd, nb_scores_nonstd = naive_bayes(standardise=False, half=half_data)
        nb_acc_nonstd = get_metric_from_scores(nb_scores_nonstd, accuracy_metric)
        print("Naive Bayes (Non-Standardised)", nb_mean_nonstd)

        mlp_mean_nonstd, mlp_scores_nonstd = mlp(standardise=False, half=half_data)
        mlp_acc_nonstd = get_metric_from_scores(mlp_scores_nonstd, accuracy_metric)
        print('Multilayer Perceptron (Non-Standardised)', mlp_mean_nonstd)

        rfc_mean_nonstd, rfc_scores_nonstd = run_rfc(standardise=False, half=half_data)
        rfc_acc_nonstd = get_metric_from_scores(rfc_scores_nonstd, accuracy_metric)
        print('Random Forest (Non-Standardised)', rfc_mean_nonstd)

        svm_mean_nonstd, svm_scores_nonstd = run_svm(standardise=False, half=half_data)
        svm_acc_nonstd = get_metric_from_scores(svm_scores_nonstd, accuracy_metric)
        print('Linear Support Vector (Non-Standardised)', svm_mean_nonstd)

        acc_list_nonstd = [cvmdl_acc_nonstd, knn_acc_nonstd, nb_acc_nonstd, mlp_acc_nonstd,
                           rfc_acc_nonstd, svm_acc_nonstd]

        # Standardized models
        cvmdl_mean, cvmdl_scores = run_covariance_model(standardise=True, half=half_data)
        cvmdl_acc = get_metric_from_scores(cvmdl_scores, accuracy_metric)
        print("Shared Covariance Model (Standardised)", cvmdl_mean)

        knn_mean, knn_scores = knn(standardise=True, half=half_data)
        knn_acc = get_metric_from_scores(knn_scores, accuracy_metric)
        print("KNN (Standardised)", knn_mean)

        nb_mean, nb_scores = naive_bayes(standardise=True, half=half_data)
        nb_acc = get_metric_from_scores(nb_scores, accuracy_metric)
        print("Naive Bayes (Standardised)", nb_mean)

        mlp_mean, mlp_scores = mlp(standardise=True, half=half_data)
        mlp_acc = get_metric_from_scores(mlp_scores, accuracy_metric)
        print('Multilayer Perceptron (Standardised)', mlp_mean)

        rfc_mean, rfc_scores = run_rfc(standardise=True, half=half_data)
        rfc_acc = get_metric_from_scores(rfc_scores, accuracy_metric)
        print('Random Forest (Standardised)', rfc_mean)

        svm_mean, svm_scores = run_svm(standardise=True, half=half_data)
        svm_acc = get_metric_from_scores(svm_scores, accuracy_metric)
        print('Linear Support Vector (Standardised)', svm_mean)

        acc_list = [cvmdl_acc, knn_acc, nb_acc, mlp_acc, rfc_acc, svm_acc]
        print(acc_list[0])
        box_plot_comparison(acc_list, acc_list_nonstd, accuracy_metric.capitalize(),
                            ["Shared Covariance Model", "KNN", "Naive Bayes",
                             "Multilayer Perceptron", "Random Forest", "Linear Support Vector"])
    elif standardise or non_standardise:
        standardisation = standardise
        if standardisation:
            text = "(Standardised)"
        else:
            text = "(Non-Standardised)"
        # models
        cvmdl_mean, cvmdl_scores = run_covariance_model(standardise=standardisation, half=half_data)
        cvmdl_acc = get_metric_from_scores(cvmdl_scores, accuracy_metric)
        print("Shared Covariance Model " + text, cvmdl_mean)

        if standardise:
            mlr_mean, mlr_scores = run_mlr_model(half=half_data)
            mlr_acc = get_metric_from_scores(mlr_scores, accuracy_metric)
            print("Multiclass Logistic Regression " + text, mlr_mean)

        knn_mean, knn_scores = knn(standardise=standardisation, half=half_data)
        knn_acc = get_metric_from_scores(knn_scores, accuracy_metric)
        print("KNN " + text, knn_mean)

        nb_mean, nb_scores = naive_bayes(standardise=standardisation, half=half_data)
        nb_acc = get_metric_from_scores(nb_scores, accuracy_metric)
        print("Naive Bayes " + text, nb_mean)

        mlp_mean, mlp_scores = mlp(standardise=standardisation, half=half_data)
        mlp_acc = get_metric_from_scores(mlp_scores, accuracy_metric)
        print('Multilayer Perceptron ' + text, mlp_mean)

        rfc_mean, rfc_scores = run_rfc(standardise=standardisation, half=half_data)
        rfc_acc = get_metric_from_scores(rfc_scores, accuracy_metric)
        print('Random Forest ' + text, rfc_mean)

        svm_mean, svm_scores = run_svm(standardise=standardisation, half=half_data)
        svm_acc = get_metric_from_scores(svm_scores, accuracy_metric)
        print('Linear Support Vector ' + text, svm_mean)

        if standardise:
            acc_list = [cvmdl_acc, knn_acc, mlr_acc, nb_acc, mlp_acc, rfc_acc, svm_acc]
            print(acc_list[0])

            box_plot(acc_list, accuracy_metric.capitalize(),
                     ["Shared Covariance Model", "KNN", "Multiclass Logistic Regression", "Naive Bayes",
                      "Multilayer Perceptron", "Random Forest", "Linear Support Vector"])

        else:
            acc_list = [cvmdl_acc, knn_acc, nb_acc, mlp_acc, rfc_acc, svm_acc]
            print(acc_list[0])

            box_plot(acc_list, accuracy_metric.capitalize(),
                     ["Shared Covariance Model", "KNN", "Naive Bayes",
                      "Multilayer Perceptron", "Random Forest", "Linear Support Vector"])
    else:
        cvmdl_mean, cvmdl_scores = run_covariance_model(half=half_data)
        cvmdl_acc = get_metric_from_scores(cvmdl_scores, accuracy_metric)
        print("Shared Covariance Model", cvmdl_mean)

        mlr_mean, mlr_scores = run_mlr_model(half=half_data)
        mlr_acc = get_metric_from_scores(mlr_scores, accuracy_metric)
        print("Multiclass Logistic Regression", mlr_mean)

        knn_mean, knn_scores = knn(half=half_data)
        knn_acc = get_metric_from_scores(knn_scores, accuracy_metric)
        print("KNN", knn_mean)

        nb_mean, nb_scores = naive_bayes(half=half_data)
        nb_acc = get_metric_from_scores(nb_scores, accuracy_metric)
        print("Naive Bayes", nb_mean)

        mlp_mean, mlp_scores = mlp(half=half_data)
        mlp_acc = get_metric_from_scores(mlp_scores, accuracy_metric)
        print('Multilayer Perceptron', mlp_mean)

        rfc_mean, rfc_scrs = run_rfc(half=half_data)
        print("Random Forest Classifier", rfc_mean)
        rfc_acc = get_metric_from_scores(rfc_scrs, accuracy_metric)

        svm_mean, svm_score = run_svm(half=half_data)
        print("Linear Support Vector Classifier", svm_mean)
        svm_acc = get_metric_from_scores(svm_score, accuracy_metric)

        acc_list = [cvmdl_acc, knn_acc, mlr_acc, nb_acc, mlp_acc, rfc_acc, svm_acc]
        box_plot(acc_list, accuracy_metric.capitalize(),
                 ["Shared Covariance Model", "KNN", "Multiclass Logistic Regression", "Naive Bayes",
                  "Multilayer Perceptron", "Random Forest", "LinearSVC"])


def run_model(model, arg_dict):
    if model == 'knn':
        if arg_dict['test_params']:
            test_knn()
        else:
            mean, scrs = knn(n_neighbors=arg_dict['n_neighbors'], half=arg_dict['half'], kfold=arg_dict['kfold'],
                             per_class_scores=arg_dict['per_class'])
            print(mean)
            if arg_dict['per_class']:
                acc = list(get_metric_from_scores(scrs, arg_dict['accuracy_metric'], per_class=arg_dict['per_class']))
                box_plot(acc, "KNN per class " + arg_dict['accuracy_metric'].capitalize(),
                         ["KNN C" + str(x) for x in [1, 2, 3]])
            else:
                acc = [get_metric_from_scores(scrs, arg_dict['accuracy_metric'])]
                box_plot(acc, "KNN " + arg_dict['accuracy_metric'].capitalize(), ['KNN'])

    elif model == 'shared_covariance_model':
        mean, scrs = run_covariance_model(half=arg_dict['half'], kfold=arg_dict['kfold'],
                                          per_class=arg_dict['per_class'])
        print(mean)
        if arg_dict['per_class']:
            acc = list(get_metric_from_scores(scrs, arg_dict['accuracy_metric'], per_class=arg_dict['per_class']))
            box_plot(acc, "Shared Covariance Model per class " + arg_dict['accuracy_metric'].capitalize(),
                     ["SCM C" + str(x) for x in [1, 2, 3]])
        else:
            acc = [get_metric_from_scores(scrs, arg_dict['accuracy_metric'])]
            box_plot(acc, "Shared Covariance Model " + arg_dict['accuracy_metric'].capitalize(), ['SCM'])

    elif model == 'multiclass_logistic_regression':
        if arg_dict['test_params']:
            optimsation_test()
        else:
            mean, scrs = run_mlr_model(half=arg_dict['half'], lrx=arg_dict['learning_rate'],
                                       n_iterx=arg_dict['no_iterations'], kfold=arg_dict['kfold'],
                                       per_class=arg_dict['per_class'])
            print(mean)
            if arg_dict['per_class']:
                acc = list(get_metric_from_scores(scrs, arg_dict['accuracy_metric'], per_class=arg_dict['per_class']))
                box_plot(acc, "Multiclass Logistic Regression per class " + arg_dict['accuracy_metric'].capitalize(),
                         ["MLR C" + str(x) for x in [1, 2, 3]])
            else:
                acc = [get_metric_from_scores(scrs, arg_dict['accuracy_metric'])]
                box_plot(acc, "Multiclass Logistic Regression " + arg_dict['accuracy_metric'].capitalize(), ['MLR'])

    elif model == 'naive_bayes':
        mean, scrs = naive_bayes(half=arg_dict['half'], kfold=arg_dict['kfold'], per_class_scores=arg_dict['per_class'])
        print(mean)
        if arg_dict['per_class']:
            acc = list(get_metric_from_scores(scrs, arg_dict['accuracy_metric'], per_class=arg_dict['per_class']))
            box_plot(acc, "Naive Bayes per class " + arg_dict['accuracy_metric'].capitalize(),
                     ["NB C" + str(x) for x in [1, 2, 3]])
        else:
            acc = [get_metric_from_scores(scrs, arg_dict['accuracy_metric'])]
            box_plot(acc, "Naive Bayes " + arg_dict['accuracy_metric'].capitalize(), ['NB'])

    elif model == 'multilayer_perceptron':
        if arg_dict['test_params']:
            test_mlp()
        else:
            mean, scrs = mlp(half=arg_dict['half'], hidden_layer_sizes=tuple(arg_dict['hidden_layer_size']),
                             activation=arg_dict['activation'], rand_state=arg_dict['rand_state'],
                             solver=arg_dict['solver'], max_iterations=arg_dict['max_iter'], kfold=arg_dict['kfold'],
                             per_class_scores=arg_dict['per_class'])
            print(mean)
            if arg_dict['per_class']:
                acc = list(get_metric_from_scores(scrs, arg_dict['accuracy_metric'], per_class=arg_dict['per_class']))
                box_plot(acc, "Multilayer Perceptron per class " + arg_dict['accuracy_metric'].capitalize(),
                         ["MP C" + str(x) for x in [1, 2, 3]])
            else:
                acc = [get_metric_from_scores(scrs, arg_dict['accuracy_metric'])]
                box_plot(acc, "Multilayer Perceptron " + arg_dict['accuracy_metric'].capitalize(), ['MP'])

    elif model == 'random_forest':
        if arg_dict['test_params']:
            run_test_params_rfc(per_class=False)
        else:
            rfc_mean, scrs = run_rfc(half=arg_dict['half'], num_estimators=arg_dict['no_estimators'],
                                     kfold=arg_dict['kfold'], max_features=arg_dict['max_features'],
                                     per_class=arg_dict['per_class'])
            print("mean=", rfc_mean)
            if arg_dict['per_class']:

                acc = list(get_metric_from_scores(scrs, arg_dict['accuracy_metric'], per_class=arg_dict['per_class']))
                print(acc)
                box_plot(acc, "Random Forest Classifier per class " + arg_dict['accuracy_metric'].capitalize(),
                         ["RFC C" + str(x) for x in [1, 2, 3]])
            else:
                acc = [get_metric_from_scores(scrs, arg_dict['accuracy_metric'])]
                box_plot(acc, "Random Forest Classifier " + arg_dict['accuracy_metric'].capitalize(), ['RFC'])

    elif model == 'linearSVC':
        if arg_dict['test_params']:
            test_svm(standardise=True, half=False, acc_param="precision", loss="squared_hinge")
        else:
            svm_mean, scrs = run_svm(half=arg_dict['half'], rand_state=arg_dict['rand_state'],
                                     max_iters=arg_dict['max_iter'], kfold=arg_dict['kfold'], c_val=arg_dict['C'],
                                     loss_func=arg_dict['loss'], per_class=arg_dict['per_class'])
            print(svm_mean)
            if arg_dict['per_class']:
                acc = list(get_metric_from_scores(scrs, arg_dict['accuracy_metric'], per_class=arg_dict['per_class']))
                box_plot(acc, "Linear SVC per class " + arg_dict['accuracy_metric'].capitalize(),
                         ["LinearSVC C" + str(x) for x in [1, 2, 3]])
            else:
                acc = [get_metric_from_scores(scrs, arg_dict['accuracy_metric'])]
                box_plot(acc, "Linear SVC " + arg_dict['accuracy_metric'].capitalize(), ['LinearSCV'])

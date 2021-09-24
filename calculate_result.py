import numpy as np
import sys


def calculate_shared_covariance_result(folds, model):
    scores = []
    for i, fold_split in enumerate(folds):
        result = model(fold_split[0], fold_split[1], fold_split[2], fold_split[3])
        scores.append(result)
        print("CV set " + str(i + 1) + " result: ", result)


def results_mean(dict_list):
    mean_dict = {}
    # print(dict_list)
    for key in dict_list[0].keys():
        mean_dict[key] = np.mean([d[key] for d in dict_list], axis=0)
    return mean_dict


# Select an accuracy metric from list of dictionaries
def get_metric_from_scores(dict_list_scores, metric_key, per_class=False):
    if per_class:
        if metric_key == 'accuracy':
            print("sorry a model's overall accuracy cannot be calculated per class")
            sys.exit()
        inter_array = [d[metric_key] for d in dict_list_scores]
        # print(inter_array)
        class1_list = [item[0] for item in inter_array]
        class2_list = [item[1] for item in inter_array]
        class3_list = [item[2] for item in inter_array]
        return class1_list, class2_list, class3_list

    return [d[metric_key] for d in dict_list_scores]

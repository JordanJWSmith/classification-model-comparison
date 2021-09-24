import numpy as np
import matplotlib.pyplot as plt


# Given a list of dictionary scores - plot box plot for chosen dict key
def box_plot(nested_list_scores, metric_name, x_labels_list):
    # plt.ion()
    fig = plt.figure()
    fig.suptitle(metric_name + " comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(nested_list_scores)
    ax.set_xticklabels(x_labels_list)
    plt.savefig('plots/' + metric_name + "_comparison.png")
    plt.draw()
    plt.pause(0.0001)


def box_plot_comparison(nested_list_scores, nested_list_scores_nonstd, metric_name, x_labels_list):
    fig = plt.figure()
    fig.suptitle("Standardised " + metric_name + " comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(nested_list_scores)
    ax.set_xticklabels(x_labels_list)
    plt.savefig("Standardised " + metric_name + "_comparison.png")

    fig2 = plt.figure()
    fig2.suptitle("Non-standardised " + metric_name + " comparison")
    ax2 = fig2.add_subplot(111)
    plt.boxplot(nested_list_scores_nonstd)
    ax2.set_xticklabels(x_labels_list)
    plt.savefig("Non-standardised " + metric_name + "_comparison.png")

    plt.show()

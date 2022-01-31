from typing import Tuple
import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray):
    """
    the first column of the output contains all examples with the label 0
    and the first row contains all examples predicted as 0 and so on
    """

    predictions = predictions.astype(np.int16)
    labels = labels.astype(np.int16)

    num_labels = np.max(labels) + 1  # labels are assumed to be the natural numbers up to 1 - the number of unique labels
    confusion = np.zeros((num_labels, num_labels))
    np.add.at(confusion, (predictions, labels), 1)

    return confusion


def precision(predictions: np.ndarray, labels: np.ndarray, target_class_label: int):
    bin_preds, bin_labels = binarize_for_target_label(predictions, labels, target_class_label)

    true_pos = num_true_positives(bin_preds, bin_labels)
    false_pos = num_false_positives(bin_preds, bin_labels)

    return true_pos/(true_pos + false_pos)


def recall(predictions: np.ndarray, labels: np.ndarray, target_class_label: int):
    bin_preds, bin_labels = binarize_for_target_label(predictions, labels, target_class_label)

    true_pos = num_true_positives(bin_preds, bin_labels)
    false_neg = num_false_negatives(bin_preds, bin_labels)

    return true_pos/(true_pos + false_neg)


def class_accuracy(predictions: np.ndarray, labels: np.ndarray, target_class_label: int):
    bin_preds, bin_labels = binarize_for_target_label(predictions, labels, target_class_label)

    num_correct = np.sum(np.where(bin_labels == bin_preds, 1, 0))
    return num_correct/len(predictions)


def all_class_accuracy(predictions: np.ndarray, labels: np.ndarray):
    return np.mean(np.where(predictions == labels, 1, 0))


def num_true_positives(bin_predictions: np.ndarray, bin_labels: np.ndarray):
    """
    Returns number of true positives

    :param bin_predictions: binarized predictions (category 1 for positive or category 0 for negative)
    :param bin_labels: binarized labels (category 1 for positive or category 0 for negative)
    :return: the number of true positive predictions made
    """

    return np.sum(np.where(bin_predictions + bin_labels == 2, 1, 0))


def num_false_positives(bin_predictions: np.ndarray, bin_labels: np.ndarray):
    """
    Returns number of false positives

    :param bin_predictions: binarized predictions (category 1 for positive or category 0 for negative)
    :param bin_labels: binarized labels (category 1 for positive or category 0 for negative)
    :return: the number of false positive predictions made
    """

    return np.sum(np.where(bin_predictions > bin_labels, 1, 0))


def num_false_negatives(bin_predictions: np.ndarray, bin_labels: np.ndarray):
    """
    Returns number of false negatives

    :param bin_predictions: binarized predictions (category 1 for positive or category 0 for negative)
    :param bin_labels: binarized labels (category 1 for positive or category 0 for negative)
    :return: the number of false negative predictions made
    """

    return np.sum(np.where(bin_predictions < bin_labels, 1, 0))


def binarize_for_target_label(predictions: np.ndarray, labels: np.ndarray, target_class_label: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms predictions and labels into binary 'does this match the class label'? arrays of 1s and 0s.

    For example, if the target class label is '2', then a prediction or label of either '0' or '1' will be transformed into
    a '0' because it does not match '2'. A prediction or label of '2' will be transformed into a '1' because it matches.

    :return: predictions with respect to the class, labels with respect to the class
    """
    binarized_predictions = np.where(predictions == target_class_label, 1, 0)
    binarized_labels = np.where(labels == target_class_label, 1, 0)

    return binarized_predictions, binarized_labels


def pr_curve(labels: np.ndarray, probabilities: np.ndarray, savepath: str):
    """
    Saves an image of a precision-recall curve to a file.
    """
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, probabilities)

    thresholds = np.concatenate([np.array([0]), thresholds], axis=0)  # prepend '0' for 'recall == 1' point

    plt.figure()
    plt.plot(recall, thresholds, '-r')
    plt.plot(recall, precision, '-b')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision (Blue) / Threshold Value (Red)")

    plt.savefig(savepath)

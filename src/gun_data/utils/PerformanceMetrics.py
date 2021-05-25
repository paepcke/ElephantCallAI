from typing import Tuple
import numpy as np


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

    true_pos = np.sum(np.where(bin_preds + bin_labels == 2, 1, 0))
    false_pos = np.sum(np.where(bin_preds > bin_labels, 1, 0))

    return true_pos/(true_pos + false_pos)


def recall(predictions: np.ndarray, labels: np.ndarray, target_class_label: int):
    bin_preds, bin_labels = binarize_for_target_label(predictions, labels, target_class_label)

    true_pos = np.sum(np.where(bin_preds + bin_labels == 2, 1, 0))
    false_neg = np.sum(np.where(bin_preds < bin_labels, 1, 0))

    return true_pos/(true_pos + false_neg)


def class_accuracy(predictions: np.ndarray, labels: np.ndarray, target_class_label: int):
    bin_preds, bin_labels = binarize_for_target_label(predictions, labels, target_class_label)

    num_correct = np.sum(np.where(bin_labels == bin_preds, 1, 0))
    return num_correct/len(predictions)


def all_class_accuracy(predictions: np.ndarray, labels: np.ndarray):
    return np.mean(np.where(predictions == labels, 1, 0))


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

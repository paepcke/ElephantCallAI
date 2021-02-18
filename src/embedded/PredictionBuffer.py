from typing import Optional
import numpy as np


class PredictionBuffer:
    """A 1-D numpy array that's also a ring buffer"""
    predictions: np.ndarray
    overlap_counts: np.ndarray

    def __init__(self, buffer_size: int):
        self.predictions = np.zeros((buffer_size,))
        self.overlap_counts = np.zeros((buffer_size,))

    def write(self, begin_idx: int, predictions_in: np.ndarray, overlap_counts_in: np.ndarray):
        if predictions_in.shape != overlap_counts_in.shape:
            raise ValueError("Cannot write if predictions and overlap")
        data_len = predictions_in.shape[0]
        if data_len + begin_idx <= self.predictions.shape[0]:
            self.predictions[begin_idx:(begin_idx + data_len)] += predictions_in
            self.overlap_counts[begin_idx:(begin_idx + data_len)] += overlap_counts_in
        else:
            first_len = self.predictions.shape[0] - begin_idx
            self.predictions[begin_idx:] += predictions_in[:first_len]
            self.overlap_counts[begin_idx:] += overlap_counts_in[:first_len]
            second_len = data_len + begin_idx - self.predictions.shape[0]
            self.predictions[:second_len] = predictions_in[first_len:]
            self.overlap_counts[:second_len] = overlap_counts_in[first_len:]

    def finalize(self, begin_idx: int, end_idx: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        if end_idx == begin_idx:
            raise ValueError("Can't finalize 0 time steps or entire buffer of time steps all in one call!"
                             " Please split up your operation across multiple subintervals.")
        pred_size = (end_idx - begin_idx) % self.predictions.shape[0]
        if out is None:
            out = np.zeros((pred_size,))
        elif out.shape[0] != pred_size:
            raise ValueError("Output array provided is the wrong size; must be size {}".format(pred_size))
        if end_idx < begin_idx:
            first_len = self.predictions.shape[0] - begin_idx
            out[0:first_len] = self.predictions[begin_idx:] / self.overlap_counts[begin_idx:]
            self.predictions[begin_idx:] = 0.
            self.overlap_counts[begin_idx:] = 0.
            out[first_len:] = self.predictions[:end_idx] / self.overlap_counts[:end_idx]
            self.predictions[:end_idx] = 0.
            self.overlap_counts[:end_idx] = 0.
        else:
            out[:] = self.predictions[begin_idx:end_idx] / self.overlap_counts[begin_idx:end_idx]
            self.predictions[begin_idx:end_idx] = 0
            self.overlap_counts[begin_idx:end_idx] = 0
        out[:] = self.sigmoid(out)
        return out

    def sigmoid(self, arg: np.ndarray):
        exp_arg = np.exp(arg)
        return exp_arg/(exp_arg + 1)

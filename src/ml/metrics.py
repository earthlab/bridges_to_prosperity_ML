"""
Contains classes and functions for calculating inference accuracy metrics

Copyright 2023 by Erick Verleye, CU Boulder Earth Lab.
"""

import warnings
from typing import Union, List

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
from pandas.errors import DtypeWarning
from sklearn import metrics

# Not great but sub-setting column could be any type. Probably a string, which for some reason numba is having trouble
# with.
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

# Don't care about mixed column type
warnings.simplefilter('ignore', category=DtypeWarning)


@nb.jit(parallel=True)
def is_in_set_pnb(a, b):
    """
    Credit:
        https://stackoverflow.com/questions/62007409/is-there-method-faster-than-np-isin-for-large-array
    """
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)


# TODO: Things could be sped up by leveraging past calculations / adding more input parameters. The pieces are here
#  though
# TODO: Needs unit tests


class Metrics:
    """
    Contains methods for calculating metrics (true positive, true negative, false positive, false negative, etc.) of the
     model inference results.
    """
    def __init__(self, inference_results_set: str, validation_set: str, subset_key: str = 'tile',
                 confidence_column: str = 'conf', prediction_column: str = 'pred', validation_column: str = 'is_bridge',
                 confidence: float = 0.834):
        """
        Initializes DataFrames for inference result set and validation set. Subsets inference results to only the rows
        in the validation set. 'subset_key' is used indicate the column that should be used to match columns in both
        datasets.
        Args:
            inference_results_set (str): Path to the inference results file. Must be able to be read in as a Pandas
             DataFrame.
            validation_set (str): Path to the validation dataset file. Must be able to be read in as a Pandas DataFrame.
            subset_key (str): Default 'tile'. Used to match rows in the validation and inference results DataFrames.
            confidence_column (str): Default 'conf'. Name of column in inference results DataFrame with confidence
                scores
            prediction_column (str): Default 'pred'. Name of the column in inference results DataFrame with prediction
            validation_column (str): Default 'is_bridge'. Name of the column in validation set with target value
            (1 for bridge / 0 no_bridge)
            confidence (float): Value of confidence threshold. Predictions with confidence lower than this will be made
            0 or 'no_bridge' in calculations. Can be overriden when calling methods by supplying confidence parameter.
        """
        self._inference_results_set = pd.read_csv(inference_results_set)
        self._validation_set = pd.read_csv(validation_set)

        self._confidence = confidence
        self._confidence_column = confidence_column
        self._prediction_column = prediction_column
        self._validation_column = validation_column

        self._inference_subset = self._subset(subset_key)
        self._predictions_above_confidence = self._calc_predictions_above_confidence(self._confidence)

    # Properties
    @property
    def inference_results_set(self) -> pd.DataFrame:
        return self._inference_results_set

    @property
    def validation_set(self) -> pd.DataFrame:
        return self._validation_set

    @property
    def inference_subset(self) -> pd.DataFrame:
        return self._inference_subset

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, confidence: float) -> None:
        if not 0 < confidence < 1.0:
            raise ValueError('Confidence must be between 0 and 1.0')
        self._confidence = confidence

    @confidence.getter
    def confidence(self) -> float:
        return self._confidence

    @property
    def confidence_column(self) -> str:
        return self._confidence_column

    @confidence_column.setter
    def confidence_column(self, column: str) -> None:
        if not isinstance(column, str):
            raise ValueError('Column name must be a string')
        self._confidence_column = column

    @property
    def prediction_column(self) -> str:
        return self._prediction_column

    @prediction_column.setter
    def prediction_column(self, column: str) -> None:
        if not isinstance(column, str):
            raise ValueError('Column name must be a string')
        self._prediction_column = column

    @property
    def validation_column(self) -> str:
        return self._validation_column

    @validation_column.setter
    def validation_column(self, column: str) -> None:
        if not isinstance(column, str):
            raise ValueError('Column name must be a string')
        self._validation_column = column

    # Private methods

    def _calc_predictions_above_confidence(self, confidence: float) -> List[int]:
        """
        Returns a list of predictions where 1 is value only if previous value was True and the confidence is above the
        threshold. The rest of the values will be 0.
        Args:
            confidence (float): Confidence threshold between 0 and 1.0. Only values above the threshold will remain True
        """
        return [
            1 if self._inference_subset[self._confidence_column].iloc[i] >= confidence and val else 0
            for i, val in enumerate(self._inference_subset[self._prediction_column])
        ]

    def _subset(self, key: str) -> pd.DataFrame:
        """
        Subsets the inference results dataframe to just the rows in the validation set. Rows are matched using the key
        parameter.
        Args:
            key (str): Name of column used for matching rows between validation and inference results DataFrames.
        """
        prediction_key_array = self._inference_results_set[key].to_numpy()
        is_in_result = is_in_set_pnb(prediction_key_array,
                                     self._validation_set[key].to_numpy()); prediction_key_array[is_in_result]

        return self._inference_results_set.iloc[np.where(is_in_result)]

    def _confidence_check(self, confidence: Union[None, float]) -> List[int]:
        if confidence is not None:
            if not 0 < confidence < 1.0:
                raise ValueError('Confidence must be between 0 and 1')
            return self._calc_predictions_above_confidence(confidence)
        else:
            return self._predictions_above_confidence

    # Public methods

    def calculate_true_positives(self, confidence: Union[float, None] = None) -> int:
        """
        Counts the number of predictions that were marked as bridge in inference results set that were also marked as
        bridge in validation set.
        Args:
            confidence (float): When set to default value of None, the predictions above confidence values calculated on
             instantiation will be used. Otherwise, predictions above confidence will be recalculated for result.
        Returns:
            true_positives (int): Number of predictions that were marked as bridge in inference results set that were
             also marked as bridge in validation set.
        """
        predictions_above_confidence = self._confidence_check(confidence)

        true_positives = np.logical_and(self._validation_set[self._validation_column].to_numpy(),
                                        predictions_above_confidence)

        return np.count_nonzero(true_positives)

    def calculate_true_negatives(self) -> int:
        """
        Counts the number of predictions that were marked as no_bridge in inference results set that were also marked as
        no_bridge in validation set.
        Returns:
            true_negatives (int): Number of predictions that were marked as no_bridge in inference results set that were
             also marked as no_bridge in validation set.
        """
        true_negatives = np.logical_and(np.logical_not(self._validation_set[self._validation_column].to_numpy()),
                                        np.logical_not(self._inference_subset[self._prediction_column].to_numpy()))

        return np.count_nonzero(true_negatives)

    def calculate_false_positives(self, confidence: Union[None, float] = None) -> int:
        """
        Counts the number of predictions that were marked as bridge in inference results set that were marked as
        no_bridge in validation set.
        Args:
            confidence (float): When set to default value of None, the predictions above confidence values calculated on
             instantiation will be used. Otherwise, predictions above confidence will be recalculated for result.
        Returns:
            false_positives (int): Number of predictions that were marked as bridge in inference results set that were
             marked as no_bridge in validation set.
        """
        predictions_above_confidence = self._confidence_check(confidence)

        false_positives = np.logical_and(np.logical_not(self._validation_set[self._validation_column].to_numpy()),
                                         predictions_above_confidence)

        return np.count_nonzero(false_positives)

    def calculate_false_negatives(self) -> int:
        """
        Counts the number of predictions that were marked as no_bridge in inference results set that were not marked as
        bridge in validation set.
        Returns:
            false_negatives (int): Number of predictions that were marked as no_bridge in inference results set that
            were not marked as bridge in validation set.
        """
        false_negatives = np.logical_and(self._validation_set[self._validation_column].to_numpy(),
                                         np.logical_not(self._inference_subset[self._prediction_column].to_numpy()))
        return np.count_nonzero(false_negatives)

    def calculate_precision(self, confidence: Union[None, float] = None) -> float:
        """
        Calculates precision of inference results defined as true_positives / (true_positives + false_positives).
        Args:
            confidence (float): When set to default value of None, the predictions above confidence values calculated on
             instantiation will be used. Otherwise, predictions above confidence will be recalculated for result.
        Returns:
            precision (float): true_positives / (true_positives + false_positives)
        """
        true_positive = self.calculate_true_positives(confidence)
        false_positive = self.calculate_false_positives(confidence)

        # Protect divide by zero
        if true_positive == 0:
            return 0

        return true_positive / (true_positive + false_positive)

    def calculate_recall(self, confidence: Union[None, float] = None) -> float:
        """
        Calculates recall of inference results defined as true_positives / (true_positives + false_negatives).
        Args:
            confidence (float): When set to default value of None, the predictions above confidence values calculated on
             instantiation will be used. Otherwise, predictions above confidence will be recalculated for result.
        Returns:
            recall (float): true_positives / (true_positives + false_negatives)
        """
        true_positive = self.calculate_true_positives(confidence)
        false_negative = self.calculate_false_negatives()

        # Protect divide by zero
        if true_positive == 0:
            return 0

        return true_positive / (true_positive + false_negative)

    def calculate_f1(self, confidence: Union[None, float] = None) -> float:
        """
        Calculates f1 of inference results defined as precision + recall.
        Args:
            confidence (float): When set to default value of None, the predictions above confidence values calculated on
             instantiation will be used. Otherwise, predictions above confidence will be recalculated for result.
        Returns:
            recall (float): precision + recall
        """
        precision = self.calculate_precision(confidence)
        recall = self.calculate_recall(confidence)

        return precision + recall

    def plot_roc(self, outpath: str = None):
        positive_confidences = [1-conf if not self._inference_subset[self._prediction_column].iloc[i] else conf for
                                i, conf in enumerate(self._inference_subset[self._confidence_column])]

        fpr, tpr, thresholds = metrics.roc_curve(self._validation_set[self._validation_column], positive_confidences)

        gmean = np.sqrt(tpr * (1 - fpr))

        # Find the optimal threshold
        index = np.argmax(gmean)
        thresholdOpt = round(thresholds[index], ndigits=4)
        gmeanOpt = round(gmean[index], ndigits=4)
        fprOpt = round(fpr[index], ndigits=4)
        tprOpt = round(tpr[index], ndigits=4)
        print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
        print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')

        plt.title('Receiver operating characteristic (ROC) curve')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')

        # Save or show the plot![](../../../../../../../var/folders/94/t8rm1cdd27d7b64yjmvykh4h0000gp/T/TemporaryItems/NSIRD_screencaptureui_WZ6T3R/Screenshot 2023-04-11 at 2.36.28 PM.png)
        if outpath is not None:
            plt.savefig(outpath)
        else:
            plt.show()

        return fpr, tpr, thresholds

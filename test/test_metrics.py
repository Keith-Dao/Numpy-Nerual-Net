"""
This module tests the metrics module.
"""
import math

import numpy as np
import pytest

from src.metrics import (
    accuracy,
    f1_score,
    get_new_confusion_matrix,
    add_to_confusion_matrix,
    precision,
    recall
)
from . import FLOAT_TOLERANCE


class TestConfusionMatrix:
    """
    Confusion matrix function tester.
    """
    @pytest.mark.parametrize("size, expected", [
        (3, np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    ])
    def test_create_confusion_matrix(self, size, expected):
        """
        Tests the create confusion matrix method.
        """
        assert np.array_equal(get_new_confusion_matrix(size), expected)

    @pytest.mark.parametrize(
        "confusion_matrix, predictions, actual, expected",
        [
            (np.zeros((3, 3)), [0, 1, 2], [0, 1, 2], np.eye(3)),
            (
                np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]]),
                [1, 2, 0, 0],
                [0, 1, 2, 0],
                np.array([[1, 1, 2], [2, 0, 0], [0, 1, 0]]),
            )
        ])
    def test_add_to_confusion_matrix(
        self,
        confusion_matrix,
        predictions,
        actual,
        expected
    ):
        """
        Tests the add_to_confusion_matrix method.
        """
        add_to_confusion_matrix(confusion_matrix, predictions, actual)
        assert np.array_equal(confusion_matrix, expected)

    def test_add_to_confusion_matrix_error(self):
        """
        Tests the add_to_confusion_matrix method with different class lengths.
        """
        with pytest.raises(ValueError):
            add_to_confusion_matrix(np.zeros((3, 3)), [0, 1, 2], [1])


class TestAccuracy:
    """
    Accuracy tester.
    """
    @pytest.mark.parametrize("confusion_matrix, expected", [
        (np.array([[5, 0, 0], [0, 2, 0], [0, 0, 10]]), 1),
        (np.array([[4, 4, 2], [0, 2, 0], [3, 2, 5]]), 0.5),
        (np.array([[20, 1, 60], [29, 13, 2], [32, 6, 34]]), 0.340101522843)
    ])
    def test_accuracy(self, confusion_matrix, expected):
        """
        Tests accuracy.
        """
        assert math.isclose(
            accuracy(confusion_matrix),
            expected,
            abs_tol=FLOAT_TOLERANCE
        )


class TestPrecision:
    """
    Precision tester.
    """
    @pytest.mark.parametrize("confusion_matrix, expected", [
        (np.array([[3, 2], [1, 4]]), [0.6, 0.8]),
        (np.array([[4, 4, 2], [0, 2, 0], [3, 2, 5]]), [0.4, 1, 0.5]),
        (
            np.array([[20, 1, 60], [29, 13, 2], [32, 6, 34]]),
            [0.24691358024691357, 0.29545454545454547, 0.4722222222222222]
        ),
        (
            np.array([
                [50, 3, 0, 0],
                [26, 8, 0, 1],
                [20, 2, 4, 0],
                [12, 0, 0, 1]
            ]),
            [
                0.9433962264150944,
                0.22857142857142856,
                0.15384615384615385,
                0.07692307692307693
            ]
        )
    ])
    def test_precision(self, confusion_matrix, expected):
        """
        Tests precision.
        """
        assert np.allclose(
            precision(confusion_matrix),
            expected,
            atol=FLOAT_TOLERANCE
        )


class TestRecall:
    """
    Recall tester.
    """
    @pytest.mark.parametrize("confusion_matrix, expected", [
        (np.array([[3, 2], [1, 4]]), [0.75, 0.666666]),
        (
            np.array([[4, 4, 2], [0, 2, 0], [3, 2, 5]]),
            [0.5714285714285714, 0.25, 0.7142857142857143]
        ),
        (
            np.array([[20, 1, 60], [29, 13, 2], [32, 6, 34]]),
            [0.24691358024691357, 0.65, 0.3541666666666667]
        ),
        (
            np.array([
                [50, 3, 0, 0],
                [26, 8, 0, 1],
                [20, 2, 4, 0],
                [12, 0, 0, 1]
            ]),
            [
                0.46296296296296297,
                0.6153846153846154,
                1,
                0.5
            ]
        )
    ])
    def test_recall(self, confusion_matrix, expected):
        """
        Tests recall.
        """
        assert np.allclose(
            recall(confusion_matrix),
            expected,
            atol=FLOAT_TOLERANCE
        )


class TestF1Score:
    """
    F1 score tester.
    """
    @pytest.mark.parametrize("confusion_matrix, expected", [
        (np.array([[3, 2], [1, 4]]), [0.66666667, 0.72727273]),
        (
            np.array([[4, 4, 2], [0, 2, 0], [3, 2, 5]]),
            [0.47058824, 0.4, 0.58823529]
        ),
        (
            np.array([[20, 1, 60], [29, 13, 2], [32, 6, 34]]),
            [0.24691358, 0.40625, 0.4047619]
        ),
        (
            np.array([
                [50, 3, 0, 0],
                [26, 8, 0, 1],
                [20, 2, 4, 0],
                [12, 0, 0, 1]
            ]),
            [
                0.6211180124223602,
                0.3333333333333333,
                0.2666666666666667,
                0.13333333333333336
            ]
        )
    ])
    def test_f1_score(self, confusion_matrix, expected):
        """
        Tests f1_score.
        """
        assert np.allclose(
            f1_score(confusion_matrix),
            expected,
            atol=FLOAT_TOLERANCE
        )

"""
This module tests the metrics module.
"""

import numpy as np
import pytest

from src.metrics import (
    get_new_confusion_matrix,
    add_to_confusion_matrix
)


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

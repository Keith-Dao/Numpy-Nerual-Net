"""
This module tests the model module.
"""
import filecmp
import math
import pathlib

import numpy as np
from numpy.typing import NDArray
import pytest
from src.activation_functions import ReLU

from src.cross_entropy_loss import CrossEntropyLoss
from src.linear import Linear
from src.model import Model
from . import FLOAT_TOLERANCE


# pylint: disable=protected-access, invalid-name, too-many-public-methods
# pylint: disable=unused-argument, too-many-arguments, line-too-long
# pylint: disable=too-few-public-methods
# pyright: reportGeneralTypeIssues=false
class TestModel:
    """
    Model tester.
    """

    # region Fixtures
    @pytest.fixture
    def layers(self) -> list[Linear]:
        """
        Gets the list of linear layers.

        Returns:
            List of linear layers.
        """
        return [
            Linear(
                4, 3,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size)
            ),
            Linear(
                3, 2,
                weight_init=lambda size: np.ones(shape=size),
                bias_init=lambda size: np.ones(shape=size),
                activation=ReLU
            )
        ]

    @pytest.fixture
    def loss(self) -> CrossEntropyLoss:
        """
        Gets a cross-entropy loss instance with a sum reduction.

        Returns:
            Cross-entropy loss instance with a sum reduction.
        """
        return CrossEntropyLoss("sum")

    @pytest.fixture
    def model(self, layers, loss) -> Model:
        """
        Gets the model.

        Returns:
            The model.
        """
        return Model(layers, loss)

    @pytest.fixture(scope="class")
    def json_file(self, tmp_path_factory):
        """
        Writes the JSON format of the model to a file.

        Returns:
            Path to the file with the JSON data.
        """
        file_path = tmp_path_factory.getbasetemp() / "true_model.json"
        with open(file_path, "w", encoding="UTF-8") as file:
            file.write('{"class": "Model", "layers": [{"class": "Linear", "weight": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], "bias": [1.0, 1.0, 1.0], "activation_function": "NoActivation"}, {"class": "Linear", "weight": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], "bias": [1.0, 1.0], "activation_function": "ReLU"}], "loss": {"class": "CrossEntropyLoss", "reduction": "sum"}, "epochs": 0, "train_history": [], "validation_history": []}')  # noqa: E501
        return file_path

    @pytest.fixture(scope="class")
    def pkl_file(self, tmp_path_factory):
        """
        Writes the pickled format of the model to a file.

        Returns:
            Path to the file with the pickle data.
        """
        file_path = tmp_path_factory.getbasetemp() / "true_model.pkl"
        with open(file_path, "wb") as file:
            file.write(b'\x80\x04\x95\xd4\x01\x00\x00\x00\x00\x00\x00}\x94(\x8c\x05class\x94\x8c\x05Model\x94\x8c\x06layers\x94]\x94(}\x94(h\x01\x8c\x06Linear\x94\x8c\x06weight\x94]\x94(]\x94(G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00e]\x94(G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00e]\x94(G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00ee\x8c\x04bias\x94]\x94(G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00e\x8c\x13activation_function\x94\x8c\x0cNoActivation\x94u}\x94(h\x01h\x06h\x07]\x94(]\x94(G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00e]\x94(G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00eeh\x0c]\x94(G?\xf0\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00eh\x0e\x8c\x04ReLU\x94ue\x8c\x04loss\x94}\x94(h\x01\x8c\x10CrossEntropyLoss\x94\x8c\treduction\x94\x8c\x03sum\x94u\x8c\x06epochs\x94K\x00\x8c\rtrain_history\x94]\x94\x8c\x12validation_history\x94]\x94u.')  # noqa: E501
        return file_path

    @pytest.fixture
    def data(self) -> tuple[NDArray, list[int]]:
        """
        Sample data.

        Returns:
            The input values and the associated labels.
        """
        return (
            np.array([
                [4,  -3,   2,   4],
                [6,  -3,   6,   1],
                [5,   9,   8,   3],
                [8, -10,   8,  -7],
                [0,   3,   7,   5],
                [7,  -6,   8,   8],
                [1, -10,  -7,   5],
                [-6,   5,   4,  -9],
                [4,  -3,   8,   6],
                [5,  -9,   2,   1]
            ], dtype=float),
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0]
        )

    @pytest.fixture
    def mock_loader(self, data, request):
        """
        Mock data loader.

        Returns:
            A mock data loader with the requested data split.
        """
        split = request.param

        class MockIter:
            """
            Mock dataset iterator.
            """

            def __init__(self, X, y, batch_size) -> None:
                self.i = 0
                self.X = X
                self.y = y
                self.batch_size = batch_size

            def __iter__(self):
                return self

            def __next__(self):
                if self.i >= len(self.X):
                    raise StopIteration

                result = (
                    self.X[self.i: self.i + self.batch_size],
                    self.y[self.i: self.i + self.batch_size]
                )
                self.i += self.batch_size
                return result

            def __len__(self):
                return math.ceil(len(self.X) / self.batch_size)

        class MockLoader:
            """
            Mock data loader.
            """

            def __init__(self, split) -> None:
                X, y = data
                train_size = int(split * len(X))
                self.train_X = X[:train_size]
                self.train_y = y[:train_size]
                self.val_X = X[train_size:]
                self.val_y = y[train_size:]

            def __call__(self, type_, batch_size):
                if type_ == "train":
                    return MockIter(self.train_X, self.train_y, batch_size)
                if type_ == "test":
                    return MockIter(self.val_X, self.val_y, batch_size)
                raise ValueError

        return MockLoader(split)
    # endregion Fixtures

    # region Init tests
    @ pytest.mark.parametrize("kwargs", [
        {},
        {"total_epochs": 10},
        {"train_history": [1, 2, 3]},
        {"validation_history": [1, 2, 3]},
        {
            "total_epochs": 20,
            "train_history": [1, 2, 3],
            "validation_history": [1, 2, 3]
        }
    ])
    def test_init(self, layers, loss, kwargs):
        """
        Tests model init.
        """

        model = Model(layers, loss, **kwargs)
        assert model.layers is layers
        assert model.loss is loss
        assert model.eval is False
        assert model.total_epochs == kwargs.get("total_epochs", 0)
        assert model.train_history == kwargs.get("train_history", [])
        assert model.validation_history == kwargs.get("validation_history", [])

    @ pytest.mark.parametrize("kwargs, exception", [
        ({"total_epochs": 1.23}, TypeError),
        ({"total_epochs": []}, TypeError),
        ({"total_epochs": "Test"}, TypeError),
        ({"total_epochs": -1}, ValueError),
        ({"total_epochs": -100}, ValueError),
        ({"train_history": 1}, TypeError),
        ({"train_history": 1.23}, TypeError),
        ({"train_history": "test"}, TypeError),
        ({"train_history": [[1, -32, 3.2]]}, TypeError),
        ({"train_history": ["1", "2"]}, TypeError),
        ({"validation_history": 1}, TypeError),
        ({"validation_history": 1.23}, TypeError),
        ({"validation_history": "test"}, TypeError),
        ({"validation_history": [[1, -32, 3.2]]}, TypeError),
        ({"validation_history": ["1", "2"]}, TypeError),
    ])
    def test_init_with_invalid_values(self, layers, loss, kwargs, exception):
        """
        Test init with invalid values.
        """
        with pytest.raises(exception):
            Model(layers, loss, **kwargs)
    # endregion Init tests

    # region Static method tests

    @ pytest.mark.parametrize("history, minibatches, expected_result", [
        ([1, 2, 3], 3, 2),
        ([1, 2, 3], 2, 2.5),
        ([1, 2, 3], 1, 3),
        ([5, 2, 4, -2], 2, 1),
        ([5, -2, 4, -2], 3, 0)
    ])
    def test_calculate_mean_epoch_loss(
        self,
        history,
        minibatches,
        expected_result
    ):
        """
        Test the calculate_mean_epoch_loss method.
        """
        assert math.isclose(
            Model.calculate_mean_epoch_loss(history, minibatches),
            expected_result,
            abs_tol=FLOAT_TOLERANCE
        )
    # endregion Static method tests

    # region Property tests
    # region Evaluation mode tests
    def test_evaluation_mode(self, model):
        """
        Tests setting the evaluation mode of the model.
        """
        def check_all_equal(mode: bool):
            assert model.eval == mode
            assert all(layer.eval == mode for layer in model._layers)

        check_all_equal(False)
        model.eval = False
        check_all_equal(False)
        model.eval = True
        check_all_equal(True)

    @ pytest.mark.parametrize("eval_", [
        1, 1.123, [], {}, "True"
    ])
    def test_evaluation_mode_invalid_type(self, model, eval_):
        """
        Tests setting the evaluation mode of the model with an invalid type.
        """
        with pytest.raises(TypeError):
            model.eval = eval_
    # endregion Evaluation mode tests

    # region Layers tests
    @ pytest.mark.parametrize("layers_", [
        [Linear(1, 2), Linear(2, 3), Linear(3, 5)],
        [Linear(1, 2)]
    ])
    def test_layers(self, model, layers_):
        """
        Tests setting the model's layers.
        """
        model.layers = layers_
        assert all(
            layer == other
            for layer, other in zip(model.layers, layers_)
        )

    @ pytest.mark.parametrize("layers_", [
        1, 1.123, {}, "True", True, Linear(1, 2), [[Linear(1, 2)]]
    ])
    def test_layers_invalid_type(self, model, layers_):
        """
        Tests setting the model's layers with an invalid type.
        """
        with pytest.raises(TypeError):
            model.layers = layers_

    def test_layers_empty_list(self, model):
        """
        Tests setting the model's layers with an empty list.
        """
        with pytest.raises(ValueError):
            model.layers = []
    # endregion Layers tests

    # region Loss tests
    def test_loss(self, model):
        """
        Test setting the model's loss.
        """
        loss = CrossEntropyLoss()
        model.loss = loss
        assert model.loss is loss

    @ pytest.mark.parametrize("loss_", [
        1, 1.23, [], "Test"
    ])
    def test_loss_with_invalid_type(self, model, loss_):
        """
        Test setting the model's loss.
        """
        with pytest.raises(TypeError):
            model.loss = loss_
    # endregion Loss tests

    # region Total epochs tests
    @ pytest.mark.parametrize("total_epochs", [
        0, 1, 20, 1000
    ])
    def test_total_epochs(self, model, total_epochs):
        """
        Test setting the model's total epochs.
        """
        model.total_epochs = total_epochs
        assert model.total_epochs == total_epochs

    @ pytest.mark.parametrize("total_epochs", [
        1.23, [], "Test"
    ])
    def test_total_epochs_with_invalid_type(self, model, total_epochs):
        """
        Test setting the model's total epochs with an invalid type.
        """
        with pytest.raises(TypeError):
            model.total_epochs = total_epochs

    @ pytest.mark.parametrize("total_epochs", [
        -1, -100
    ])
    def test_total_epochs_with_negative_numbers(self, model, total_epochs):
        """
        Test setting the model's total epochs with negative numbers.
        """
        with pytest.raises(ValueError):
            model.total_epochs = total_epochs
    # endregion Total epochs tests

    # region Train history tests
    @ pytest.mark.parametrize("train_history", [
        [1, 32, 42],
        [],
        [32.2, -32, 3]
    ])
    def test_train_history(self, model, train_history):
        """
        Test setting the model's train history.
        """
        model.train_history = train_history
        assert model.train_history == train_history

    @ pytest.mark.parametrize("train_history", [
        1,
        1.23,
        "test",
        [[1, -32, 3.2]],
        ["1", "2"]
    ])
    def test_train_history_with_invalid_type(self, model, train_history):
        """
        Test setting the model's train history with invalid type.
        """
        with pytest.raises(TypeError):
            model.train_history = train_history
    # endregion Train history tests

    # region Validation history tests
    @ pytest.mark.parametrize("validation_history", [
        [1, 32, 42],
        [],
        [32.2, -32, 3]
    ])
    def test_validation_history(self, model, validation_history):
        """
        Test setting the model's validation history.
        """
        model.validation_history = validation_history
        assert model.validation_history == validation_history

    @ pytest.mark.parametrize("validation_history", [
        1,
        1.23,
        "test",
        [[1, -32, 3.2]],
        ["1", "2"]
    ])
    def test_validation_history_with_invalid_type(
        self,
        model,
        validation_history
    ):
        """
        Test setting the model's validation history with invalid type.
        """
        with pytest.raises(TypeError):
            model.validation_history = validation_history
    # endregion Validation history tests
    # endregion Property tests

    # region Load tests
    @ pytest.mark.parametrize("attributes", [
        {},
        {"total_epochs": 10},
        {"train_history": [1, 2, 3]},
        {"validation_history": [1, 2, 3]},
        {
            "total_epochs": 10,
            "train_history": [1, 2, 3],
            "validation_history": [1, 2, 3]
        },
    ])
    def test_from_dict(self, model, attributes):
        """
        Tests from_dict method.
        """
        for attribute, value in attributes.items():
            setattr(model, attribute, value)

        attributes["class"] = "Model"
        attributes["layers"] = [layer.to_dict() for layer in model.layers]
        attributes["loss"] = model.loss.to_dict()

        new_model = Model.from_dict(attributes)

        assert model.layers == new_model.layers
        assert model.loss == new_model.loss
        assert model.total_epochs == new_model.total_epochs
        assert model.train_history == new_model.train_history
        assert model.validation_history == new_model.validation_history

    def test_from_dict_with_invalid_class_name(self, model):
        """
        Tests from_dict method with invalid class name.
        """
        attributes = {
            "class": "INVALID",
            "layers": [layer.to_dict() for layer in model.layers],
            "loss": model.loss.to_dict()
        }

        with pytest.raises(ValueError):
            Model.from_dict(attributes)

    @ pytest.mark.parametrize("file_path", [
        "json_file", "pkl_file"
    ])
    @ pytest.mark.parametrize("to_string", [True, False])
    def test_load(self, model, file_path, to_string, request):
        """
        Tests the load method.
        """
        file_path = request.getfixturevalue(file_path)
        if to_string:
            file_path = str(file_path)
        new_model = Model.load(file_path)

        assert model.layers == new_model.layers
        assert model.loss == new_model.loss
        assert model.total_epochs == new_model.total_epochs
        assert model.train_history == new_model.train_history
        assert model.validation_history == new_model.validation_history

    @ pytest.mark.parametrize("file_path", [
        1, 1.13, [], {}
    ])
    def test_load_invalid_type(self, file_path):
        """
        Tests the load method with an invalid type.
        """
        with pytest.raises(TypeError):
            Model.load(file_path)

    @ pytest.mark.parametrize("file_path", [
        "/tmp/test.test",
        pathlib.Path("/tmp/test.test")
    ])
    def test_load_invalid_extension(self, file_path):
        """
        Tests the load method with an invalid extension.
        """
        with pytest.raises(ValueError):
            Model.load(file_path)
    # endregion Load tests

    # region Save tests
    def test_to_dict(self, model):
        """
        Tests the to_dict method.
        """
        assert model.to_dict() == {
            "class": "Model",
            "layers": [
                {
                    "class": "Linear",
                    "weight": [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0]
                    ],
                    "bias": [1.0, 1.0, 1.0],
                    "activation_function": "NoActivation"
                }, {
                    "class": "Linear",
                    "weight": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    "bias": [1.0, 1.0],
                    "activation_function": "ReLU"
                }
            ],
            "loss": {"class": "CrossEntropyLoss", "reduction": "sum"},
            "epochs": 0,
            "train_history": [],
            "validation_history": []
        }

    @ pytest.mark.parametrize("file_path", [
        "json_file", "pkl_file"
    ])
    @ pytest.mark.parametrize("to_string", [True, False])
    def test_save(self, model, file_path, to_string, tmp_path, request):
        """
        Tests the save method.
        """
        file_path = request.getfixturevalue(file_path)
        new_file_path = tmp_path / f"model.{file_path.suffix}"
        if to_string:
            new_file_path = str(new_file_path)
        model.save(new_file_path)

        assert filecmp.cmp(str(file_path), str(new_file_path))

        # Clean up the created file
        pathlib.Path(new_file_path).unlink()

    @ pytest.mark.parametrize("file_path", [
        1, 1.23, []
    ])
    def test_save_with_invalid_type(self, model, file_path):
        """
        Tests the save method with an invalid type.
        """
        with pytest.raises(TypeError):
            model.save(file_path)

    @ pytest.mark.parametrize("file_path", [
        "/tmp/test.test",
        pathlib.Path("/tmp/test.test")
    ])
    def test_save_with_invalid_extension(self, model, file_path):
        """
        Tests the save method with an invalid extension.
        """
        with pytest.raises(ValueError):
            model.save(file_path)
    # endregion Save tests

    # region Forward pass tests
    def test_forward(self, model, data):
        """
        Tests the forward pass.
        """
        X, *_ = data
        assert np.array_equal(
            model(X),
            np.array([
                [25., 25.],
                [34., 34.],
                [79., 79.],
                [1.,  1.],
                [49., 49.],
                [55., 55.],
                [0.,  0.],
                [0.,  0.],
                [49., 49.],
                [1.,  1.]
            ])
        )
    # endregion Forward pass tests

    # region Backward pass / train tests
    @pytest.mark.parametrize("mock_loader", [1], indirect=["mock_loader"])
    def test_train_no_validation(self, model, mock_loader):
        """
        Test train for one epoch using all the data for training.
        """
        epochs = 1
        model.train(mock_loader, 1e-4, 1, epochs)
        assert model.total_epochs == epochs
        assert all(
            math.isclose(x, y, abs_tol=FLOAT_TOLERANCE)
            for x, y in zip(
                model.train_history,
                [
                    0.6931471805599453,
                    0.7064849592415489,
                    0.6809533903573027,
                    0.6930971542052906,
                    0.6262611634252911,
                    0.8183061861646324,
                    0.6931471805599453,
                    0.6931471805599453,
                    0.6380195021075894,
                    0.6932850372585229
                ]
            )
        )
        assert model.validation_history == []
        assert np.allclose(
            model.layers[0].weight,
            np.array([[0.99999876, 1.0000021,  1.00000052, 0.99999819],
                      [0.99999876, 1.0000021,  1.00000052, 0.99999819],
                      [0.99999876, 1.0000021,  1.00000052, 0.99999819]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].weight,
            np.array([[0.99806522, 0.99806522, 0.99806522],
                      [1.00193478, 1.00193478, 1.00193478]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[0].bias,
            np.array([0.99999993, 0.99999993, 0.99999993]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].bias,
            np.array([0.99991213, 1.00008787]),
            atol=FLOAT_TOLERANCE
        )

    @pytest.mark.parametrize("mock_loader", [0.7], indirect=["mock_loader"])
    def test_train_with_validation(self, model, mock_loader):
        """
        Test train for one epoch using 70% of the data for training.
        """
        epochs = 1
        model.train(mock_loader, 1e-4, 1, epochs)
        assert model.total_epochs == epochs
        assert all(
            math.isclose(x, y, abs_tol=FLOAT_TOLERANCE)
            for x, y in zip(
                model.train_history,
                [
                    0.6931471805599453,
                    0.7064849592415489,
                    0.6809533903573027,
                    0.6930971542052906,
                    0.6262611634252911,
                    0.8183061861646324,
                    0.6931471805599453
                ]
            )
        )
        assert all(
            math.isclose(x, y, abs_tol=FLOAT_TOLERANCE)
            for x, y in zip(
                model.validation_history,
                [0.6931471805599453, 0.6380195021075894, 0.6932378476943687]
            )
        )
        assert np.allclose(
            model.layers[0].weight,
            np.array([[0.99999928, 1.00000069, 1.00000002, 0.99999771],
                      [0.99999928, 1.00000069, 1.00000002, 0.99999771],
                      [0.99999928, 1.00000069, 1.00000002, 0.99999771]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].weight,
            np.array([[0.99881988, 0.99881988, 0.99881988],
                      [1.00118012, 1.00118012, 1.00118012]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[0].bias,
            np.array([1.00000001, 1.00000001, 1.00000001]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].bias,
            np.array([0.99990929, 1.00009071]),
            atol=FLOAT_TOLERANCE
        )

    @pytest.mark.parametrize("mock_loader", [0.7], indirect=["mock_loader"])
    def test_train_with_validation_multiple_epoch(
        self,
        model,
        mock_loader
    ):
        """
        Test train for three epoch using 70% of the data for training.
        """
        epochs = 3
        model.train(mock_loader, 1e-4, 1, epochs)
        assert model.total_epochs == epochs
        assert all(
            math.isclose(x, y, abs_tol=FLOAT_TOLERANCE)
            for x, y in zip(
                model.train_history,
                [
                    0.6931471805599453,
                    0.7064849592415489,
                    0.6809533903573027,
                    0.6930971542052906,
                    0.6262611634252945,
                    0.8183061861646324,
                    0.6931471805599453,
                    0.721964285866928,
                    0.6680617110920183,
                    0.596464889964369,
                    0.6930142081547784,
                    0.581318279730699,
                    0.8796890826796818,
                    0.6931471805599453,
                    0.7456241202965298,
                    0.6383687106193249,
                    0.5345381008115466,
                    0.6929423330716423,
                    0.5470199182330638,
                    0.9309717222259597,
                    0.6931471805599453
                ]
            )
        )
        assert all(
            math.isclose(x, y, abs_tol=FLOAT_TOLERANCE)
            for x, y in zip(
                model.validation_history,
                [
                    0.6931471805599453,
                    0.6380195021075894,
                    0.6932378476943687,
                    0.6931471805599453,
                    0.5961888365020174,
                    0.6933155304522359,
                    0.6931471805599453,
                    0.5640724625404642,
                    0.6933828977003443
                ]
            )
        )
        assert np.allclose(
            model.layers[0].weight,
            np.array([[0.99999946, 1.00000456, 1.00000442, 0.99998909],
                      [0.99999946, 1.00000456, 1.00000442, 0.99998909],
                      [0.99999946, 1.00000456, 1.00000442, 0.99998909]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].weight,
            np.array([[0.99711622, 0.99711622, 0.99711622],
                      [1.00288378, 1.00288378, 1.00288378]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[0].bias,
            np.array([1.00000042, 1.00000042, 1.00000042]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].bias,
            np.array([0.99976392, 1.00023608]),
            atol=FLOAT_TOLERANCE
        )

    @pytest.mark.parametrize("mock_loader", [0.7], indirect=["mock_loader"])
    def test_train_with_validation_and_larger_batch_size(
        self,
        model,
        mock_loader
    ):
        """
        Test train for one epoch using 70% of the data for training and batch
        size of 3.
        """
        epochs = 1
        model.train(mock_loader, 1e-4, 3, epochs)
        assert model.total_epochs == epochs
        assert all(
            math.isclose(x, y, abs_tol=FLOAT_TOLERANCE)
            for x, y in zip(
                model.train_history,
                [2.0794415416798357, 2.09358137165225, 0.6931471805599453]
            )
        )
        assert all(
            math.isclose(x, y, abs_tol=FLOAT_TOLERANCE)
            for x, y in zip(
                model.validation_history,
                [2.02241995571785]
            )
        )
        assert np.allclose(
            model.layers[0].weight,
            np.array([[1.00000007, 0.99999989, 1.00000085, 0.99999841],
                      [1.00000007, 0.99999989, 1.00000085, 0.99999841],
                      [1.00000007, 0.99999989, 1.00000085, 0.99999841]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].weight,
            np.array([[0.998776, 0.998776, 0.998776],
                      [1.001224, 1.001224, 1.001224]]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[0].bias,
            np.array([1.00000012, 1.00000012, 1.00000012]),
            atol=FLOAT_TOLERANCE
        )
        assert np.allclose(
            model.layers[1].bias,
            np.array([0.99990739, 1.00009261]),
            atol=FLOAT_TOLERANCE
        )
    # endregion Backward pass / train tests

    # region Built-ins tests
    @ pytest.mark.parametrize("layers_, loss_, result", [
        # Same
        (
            [
                Linear(
                    4, 3,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size)
                ),
                Linear(
                    3, 2,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size),
                    activation=ReLU
                )
            ],
            CrossEntropyLoss("sum"),
            True
        ),
        # Different layers
        (
            [
                Linear(
                    4, 3,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.zeros(shape=size)
                ),
                Linear(
                    3, 2,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size),
                    activation=ReLU
                )
            ],
            CrossEntropyLoss("sum"),
            False
        ),
        # Different loss
        (
            [
                Linear(
                    4, 3,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size)
                ),
                Linear(
                    3, 2,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size),
                    activation=ReLU
                )
            ],
            CrossEntropyLoss("mean"),
            False
        ),
        # Different amount of layers
        (
            [
                Linear(
                    4, 3,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size)
                ),
                Linear(
                    3, 2,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size),
                    activation=ReLU
                ),
                Linear(
                    2, 2,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size),
                    activation=ReLU
                )
            ],
            CrossEntropyLoss("sum"),
            False
        ),
    ])
    def test_dunder_eq(self, model, layers_, loss_, result):
        """
        Tests the __eq__ method.
        """
        other = Model(layers_, loss_)
        assert (model == other) is result

    @ pytest.mark.parametrize("other", [
        1,
        1.23,
        -1,
        [],
        {
            "layers": [
                Linear(
                    4, 3,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size)
                ),
                Linear(
                    3, 2,
                    weight_init=lambda size: np.ones(shape=size),
                    bias_init=lambda size: np.ones(shape=size),
                    activation=ReLU
                )
            ],
            "loss": CrossEntropyLoss("sum")
        }
    ])
    def test_dunder_eq_with_different_type(self, model, other):
        """
        Test the __eq__ method with different types.
        """
        assert (model == other) is False
    # endregion Built-ins tests

# Neural Net Using NumPy

This project is an implementation of a simple neural network with two hidden layers, made specifically to classify digits in the MNIST dataset. Only NumPy can be used when performing the computations required by the network.

For the complete technical overview see [link](https://github.com/Keith-Dao/Neural-Net-From-Scratch/blob/main/README.md).

## 1. Setup

To set up the project:

If you are running a linux system:

1. Alter the values of `python` and `pip` of the makefile to suit your system.
2. Run `make .env`
3. Run `. .env/bin/activate`
4. Run `make install`

Otherwise:

1. Create and activate a python virtual environment, follow [link](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) for instructions
2. Install the packages using `pip install -r requirements.txt`
3. Install the local packages using `pip install -e .`

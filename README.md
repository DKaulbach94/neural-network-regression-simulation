# neural-network-regression-simulation

This simulation was part of my master thesis.
By using the function Simulation() you can simulate a regression by a neural network
with different hyperparameters.
The main goal in this project was to understand the path which is taken in the high dimensional
parameter space of the neural network during the learning process.

The data which is used for learning and testing is from a three dimensional linear model where
x1,x2,x3 are uniformly distributed in [0,100] and y = 2x1 - 5x2 + 4x3 + n where 
n is normally distributed with mean = 0 and a standard deviation which can be choosen.

The neural network that is used for the regression is a fully-connected feedforward network
with 3 input-neurons, 3 hidden-layers with 10 neurons each and 1 output-neuron.

One key in this simulation is that it can allways create new data so no data is used twice.
Because of that overfitting does not occur so the learning process can be analysed separatly
from the effects it could cause.


Simulation prints the following measurements that are frequently made during the training process as raw data and as a plot over time:

• The empirical risk

• The euclidean length (in a high dimension) of the gradient

• The euclidean distance (in a high dimension) between the current and the previous parameter vector. It is the same like the euclidean length (in a high dimension) of the gradient times the learning rate used in this epoch.

• The angle (in a high dimension) between the current and the previous gradient. This can be useful to understand the direction the learning path is moving to. It also shows if the learning path is straight or chaotic.

• The euclidean distance (in a high dimension) between the current parameter vector and a parameter vector which was fixed at an chosen epoch. As well as the angles these distances can be important to understand the learning path

Also the initial parameter vector and the parameter vector after the learning process are printed.

Args in Simulation():

• name: Is a string. It will be the headlines for the graphs that are printed.
you can insert the name of the learning method there.

• opt_type: Is a tensorflow-train-object. By choosing ”opt type“ you
choose the training method that is used for example tf.train.AdagradOptimizer

• learning_rate: Is a float object. It determines the global learning rate.

• batch_size: Is an integer object. It determines how much training data is created.

• minibatch_size: Is an integer object. It determines how big the minibatchs are.
The number of training steps is batch_size // minibatch_size.

• test_size: Is an integer object. It determines how much data is used to estimate the risk by the empirical risk.

• ntests: Is an integer object. It determines how many measurements are made frequently during the learning process.
This can be a runtime-critical parameter.

• stddev: Is a float object. It determines the standard deviation of n. By using stddev = 0 the data can be made deterministic

• diminish: Is a boolean object. If diminish = False then opt_type will be applied with the constant learning rate learning_rate.
If diminish = True then opt_type will be applied with the diminishing learning rate learning_rate/n where n is the number of the epoch.
It is recommended to choose diminish = False.

• distnr: Is an integer and should be between 1 and ntests.
The parameter vector at the measurement with the number distnr is marked.
Since then the euclidean distance (in a high dimension) between the current paremeter vector and the marked parameter vector is measured.

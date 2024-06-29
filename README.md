# GNN_Multivariate - Data Driven Adjacency Matrix Learning
This model uses a GNN architecture for multivariate time series forecasting.\\
The adjacency matrix is learned from the data using a neural network, hence pre-defined graph topologies is not required.\n
The input for the model is a set of features at certain time points for a given number of nodes.\n
A sliding window approach is used to create the labels for training. Depending on the window size and forecast length, the data is sliced to create the training data points.\n
The model takes the following arguments:

# DeepCluster with constructive induction
The time complexity of most clustering algorithms depends on the dimensionality of the input data and thus most clustering algorithms are slow on highdimensional data. To solve this problem, we trained a deep autoencoder and used it to compress the input data into a lower dimensional space with information loss. We reimplemented and extended the DeepCluster framework proposed by Tian et al [26]. The original framework supports only K-means and GMM clusterings. We extended it with hierarchical clustering, DBSCAN, and ensemble clustering. We evaluated the clusters and interpreted the autoencoder with constructive induction. Both frameworks proved to be unsuccessful in our experiments. However, we were able to interpret the model and visualize its knowledge

## How to use the code
The code was written in python3.6.8, all dependencies are listed in requirements.txt.

Use: `python3 Main.py <Sample> <Data> <Mode> <Algorithm>`

Sample is a number between 0 and 1. Will determine the sample size. For example 0.3 means 30% of the data will be in the sample.

Data is the dataset. Supported data: `USPS, R10K, MNIST, FMNIST, IMDB, CIFAR100`.

Mode determines if we will use basic clustering, deep autoencoder or deep cluster: `CLS, DAE, DC`.

Algorithm determines which clustering algorithm will be used: `KMEANS, GMM, HIER, DBSCAN, ENSEMBLE`.

Example: `python3 Main.py 0.3 USPS DC GMM`

To get constructs use the appropriate executable file:

Use: `python3 <dataset>.py <construct_name> <labels_name>`

Example: `python3 Constructs/weights_USPS_DC_GMM9_indices_construct_0.5.csv Labels/USPS_DC_GMM_svm_labels.csv`
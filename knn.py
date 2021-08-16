import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with hamming distance """

  def __init__(self, X_train, y_train):
    """
    Initializing the KNN object

    Inputs:
    - X_train: A numpy array or pandas DataFrame of shape (num_train, D) 
    - y_train: A numpy array or pandas Series of shape (N,) containing the training labels
    """
    self.X_train = X_train
    self.y_train = y_train
    
  def fit_predict(self, X_test, k=1):
    """
    This method fits the data and predicts the labels for the given test data.
    For k-nearest neighbors fitting (training) is just 
    memorizing the training data. 
    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) 
    - k: The number of nearest neighbors.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing predicted labels
    """
    dists = self.compute_distances(X_test)
    return self.predict_labels(dists, k=k)

  def compute_distances(self, X_test):
    """
    Compute the hamming distance between each test point in X_test and each training point
    in self.X_train.

    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the hamming distance between the ith test point and the jth training
      point.
    """

    b = np.array(X_test)
    a = np.array(self.X_train)

    def ham(v1, v2):
        return (~(v1==v2)).sum()

    dists = np.array([[ham(i, j) for i in a] for j in b])
        
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing the
    predicted labels for the test data  
    """

    alpha = dists.argsort(axis=1)
    c = np.array(self.y_train)

    # get an array of the closest k labels
    preds = c[alpha[:,:k]]

    # count all the labels for each isntance, get an array of dictionaries
    preds = [{i:list(j).count(i) for i in j} for j in preds]

    # perform majority vote to get predictied labels
    y_pred = np.array([list({m: v for m, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}.keys())[0] for x in preds])
         
    return y_pred
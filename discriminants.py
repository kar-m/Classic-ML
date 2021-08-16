import numpy as np
import pandas as pd

class BinaryLDA:
  """
	Binary Linear Discriminant Analysis (LDA) classifier for classification,
	where the labels are encoded as 0s and 1s
  """
  def __init__(self):
    self.w = None
    self.t = None
	
  def get_covariance_matrix(self, X, N):
    """
    Calculate the covariance matrix for the dataset X
    """
    X = np.array(X)
    mean = X.mean(axis=0)
    a = X - mean
    cov = a.T@a
    cov /= (N-2)
    return cov

  def fit(self, X, y):
    # Separate data by class for convenience
    X1 = X[y == 0]
    X2 = X[y == 1]
    
    # Make data into numpy arrays
    X1 = np.array(X1)
    X2 = np.array(X2)
    
    N = X.shape[0]
    
    # Calculate the covariance matrices of the two datasets
    cov1 = self.get_covariance_matrix(X1, N)
    cov2 = self.get_covariance_matrix(X2, N)
    
    # cov1 and cov2 should already be normalized,
    # therefore we just add them to get sigma
    sigma = cov1 + cov2   
    
    # Calculate the mean of the two datasets
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    mean_diff = mean1 - mean2
    mean_sum = mean1 + mean2

    # Calculate the class priors
    p1 = X1.shape[0]/N
    p2 = X2.shape[0]/N

    # Get the inverse of sigma
    sigma_inv = np.linalg.inv(sigma)

    # determine the decision boundary w*x=t
    w = sigma_inv @ mean_diff
    self.w = w.T
    self.t = 0.5 * ((mean_sum.T @ sigma_inv) @ mean_diff)-np.log(p1/p2)

  def predict(self, X):
    y_pred = (self.w @ X > self.t)
    return y_pred

class LDA:
  def __init__(self):
    self.labels = None
    self.alphas = None
    self.betas = None
    self.gammas = None

  def fit(self, X, y):
    # compute means, priors for each class and
    # sigma by adding the class covariance matrices 
    def get_covariance_matrix(X, N): 
      K = len(self.labels)
      X = np.array(X)
      mean = X.mean(axis=0)
      a = X - mean
      cov = a.T @ a
      cov /= (N-K)
      return cov
  
    self.labels = np.array(pd.Series(y).unique())
    labels = self.labels
    N = X.shape[0]
    sigma = np.array([get_covariance_matrix((X[y == j]), N) for j in labels])
    sigma = sigma.sum(axis=0)
    means = np.array([X[y==j].mean(axis=0) for j in labels])
    priors = [(y==j).sum()/N for j in labels]
    # get the inverse of sigma
    sigma_inv = np.linalg.inv(sigma)

    # we need to compute the values of the 
    # discriminant functions for each class, so we will need
    # the respective coefficients to use in the 'predict' method
    
    alphas = means @ sigma_inv
    betas = (0.5*(means @ sigma_inv) @ means.T).diagonal()
    gammas = np.log(np.array(priors))
    
    self.alphas, self.betas, self.gammas = alphas, betas, gammas

  def predict(self, X):
    # calculate the delta_k for every class and every instance in X
    delta_k = (X @ self.alphas.T) - self.betas + self.gammas

    # choose the highest delta_k for every instance
    mask = delta_k.argmax(axis=1)

    y_pred = self.labels[mask]
    return y_pred


class QDA:
  """
	Quadratic Discriminant Analysis (QDA) classifier 
  for multiclass classification with arbitrary label encoding
  """
  def __init__(self):
    self.labels = None
    pass

  def fit(self, X, y):

    def get_covariance_matrix(X):
      X = np.array(X)
      N = X.shape[0]
      mean = X.mean(axis=0)
      a = X - mean
      cov = a.T @ a
      cov /= (N-1)
      return cov

    self.labels = np.array(pd.Series(y).unique())
    labels = self.labels

    N = X.shape[0]

    sigmas = np.array([get_covariance_matrix((X[y == j])) for j in labels])
    means = np.array([X[y==j].mean(axis=0) for j in labels])
    priors = np.array([(y==j).sum()/N for j in labels])
    inv_sigmas = np.array([np.linalg.inv(sigma) for sigma in sigmas])

    alphas = 0.5 * inv_sigmas
    betas = 2*alphas @ means.T
    gammas = ((means @ alphas) @ means.T).diagonal().diagonal()
    etas = np.log(priors) - 0.5 * np.log(np.linalg.det(sigmas))

    self.alphas, self.betas, self.gammas, self.etas = alphas, betas, gammas, etas

  def predict(self, X):

    a = X @ self.alphas @ X.T
    a = a.diagonal(axis1=1, axis2=2).T
    b = (X @ self.betas).diagonal(axis1=0, axis2=2)

    mask = b - a - self.gammas + self.etas
    mask = np.argmax(mask, axis=1)
    y_pred = self.labels[mask]
    return y_pred




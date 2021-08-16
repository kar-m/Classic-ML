import numpy as np

class MyNaiveBayes:
  def __init__(self, smoothing=False):
    # initialize Laplace smoothing parameter
    self.smoothing = smoothing
    
  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
    self.lbs = np.array(list(set(y_train)))
    self.feats = [np.array(list(set(j))) for j in np.array(X_train).T]
    self.feat_num = len(self.feats)
    self.priors = self.calculate_priors()
    self.likelihoods = self.calculate_likelihoods()      
      
  def predict(self, X_test):

    a = np.array(X_test)
    b = np.array(self.lbs)
    l = self.likelihoods
    alpha = np.ones((len(a), len(self.lbs)))

    for j in range(len(self.lbs)):
      lbl = self.lbs[j]
      alpha[:, j] *= self.priors[j]
      
    alpha *= 1000000

    for i in range(len(a)):
      vec = a[i]

      for j in range(len(self.lbs)):
        lbl = self.lbs[j]
        f_k = 0

        for feature_loc in vec:
          mult = l[lbl][f_k][feature_loc]
          alpha[i, j] *= mult
          f_k += 1

    mask = alpha.argmax(axis=1)
    prediction = b[mask]
     
    return prediction

  def calculate_priors(self):             
      a = np.array(self.y_train)
      priors = [np.count_nonzero(a==i)/len(a) for i in self.lbs]      
        
      return priors
  
  def calculate_likelihoods(self):
    a = np.array(self.X_train)
    b = np.array(self.y_train)
    sm = self.smoothing

    def find_likelihood(lbs, f_k, feature_loc):
      m1 = (a[:, f_k]==feature_loc)
      m2 = (b==lbs)
      ups = (m1 & m2).sum()
      alls = m2.sum()
      return (ups+sm)/(alls+sm*len(self.lbs))
    likelihoods = {lbs: {f_k: {feature_loc: find_likelihood(lbs, f_k, feature_loc) for feature_loc in self.feats[f_k]} for f_k in range(self.feat_num)} for lbs in self.lbs}
 
    return likelihoods
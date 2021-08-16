import numpy as np
import progressbar
import cvxopt

# hide cvxopt output
cvxopt.solvers.options['show_progress'] = False

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

class MyLinearRegression:

  def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
    """
    This class implements linear regression models
    Params:
    --------
    regularization - None for no regularization
                    'l2' for ridge regression
                    'l1' for lasso regression

    lam - lambda parameter for regularization in case of 
        Lasso and Ridge

    learning_rate - learning rate for gradient descent algorithm, 
                    used in case of Lasso

    tol - tolerance level for weight change in gradient descent
    """
    
    self.regularization = regularization 
    self.lam = lam 
    self.learning_rate = learning_rate 
    self.tol = tol
    self.weights = None
  
  def fit(self, X, y):
    
    X = np.array(X)

    # insert a column with all 1s in the beginning
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

    if self.regularization is None:
      self.weights = np.linalg.inv((X.T @ X))@(X.T@np.array(y))

    elif self.regularization == 'l2':
      # the case of Ridge regression
      self.weights = np.linalg.inv((X.T @ X + self.lam*np.identity(X.shape[1])))@(X.T@y)

    elif self.regularization == 'l1':
      a = np.linalg.inv((X.T @ X))@(X.T@y)
      self.weights = np.random.normal(0, 1, a.shape)

      converged = False

      # we can store the loss values to see how fast the algorithm converges
      self.loss = []

      # a counter of algorithm steps
      i = 0 

      while (not converged):
        i += 1

        # calculate the predictions in case of the weights in this stage
        y_pred = X@self.weights

        # calculate the mean squared error (loss) for the predictions
        self.loss.append((y_pred-y).T@(y_pred-y)+self.lam*np.sum(np.abs(self.weights)))

        # calculate the gradient of the objective function with respect to w
        # for the second component \sum|w_i| use np.sign(w_i) as it's derivative
        grad = -2*X.T@(y-y_pred)+self.lam*np.sign(self.weights)
        new_weights = self.weights - self.learning_rate * grad

        converged = (np.linalg.norm(new_weights-self.weights)<self.tol)
        self.weights = new_weights
      print(f'Converged in {i} steps')

  def predict(self, X):
    X = np.array(X)

    #  the feature of 1s in the beginning
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)

    # predict using the obtained weights
    return X@self.weights

def calculate_entropy(y):
  a = y.flatten()
  p = np.array([(a==i).sum() for i in set(a)])/len(a)
  log_p = np.log2(p)
  entropy = -1 * (p.T @ log_p)
  return entropy

def calculate_gini(y):
  a = y.flatten()
  p = np.array([(a==i).sum() for i in set(a)])/len(a)
  gini = 1-np.sum(p**2)
  return gini 

class DecisionNode:
  # we introduce a class for decision nodes in our tree
  # the object from this class will store all the necessary info
  # about the tree node such as 
  # the index of the best feature, which resulted in better split (feature_id)
  # the threshold with which we compare the feature values (threshold)
  # if the node is a leaf, then the label will be stored in 'value' attribute
  # if the node is not a leaf, then it has 
  # true_branch (condition is satisfied) 
  # false_branch (condition is not satisfied) subtrees 
  def __init__(self, feature_id=None, threshold=None,
                value=None, true_branch=None, false_branch=None):
      self.feature_id = feature_id          
      self.threshold = threshold          
      self.value = value                  
      self.true_branch = true_branch      
      self.false_branch = false_branch    

class RegressionTree:
  # This is the main class that recursively grows a decision tree and 
  # predicts (again recursively) according to it
  def __init__(self, impurity='entropy', min_samples_split=2,
                      min_impurity=1e-7, max_depth=float("inf")):   
    # Minimum number of samples to perform spliting
    self.min_samples_split = min_samples_split

    # The minimum impurity to perform spliting
    self.min_impurity = min_impurity

    # The maximum depth to grow the tree until
    self.max_depth = max_depth

    # Function to calculate impurity 
    self.impurity = np.var

    # Root node in dec. tree
    self.root = None  


  def calculate_purity_gain(self, y, y1, y2):
    y_imp = self.impurity(y)
    y1_imp = self.impurity(y1)
    y2_imp = self.impurity(y2)
    pure_gain = y_imp - y1_imp * len(y1)/len(y) - y2_imp * len(y2)/len(y)
    return pure_gain
    
  def divide_on_feature(self, X, y, feature_id, threshold):
    # This function divides the dataset into two parts according to the 
    # logical operation that is performed on the feature with feature_id
    # comparing against threshold
    # you should consider 2 cases: 
    # when the threshold is numeric, true cases will be feature >= threshold 
    # and when it's not, true cases will be feature == threshold  
    if isinstance(threshold, int) or isinstance(threshold, float):
        true_indices = (X[:, feature_id] >= threshold)
    else:
        true_indices = (X[:, feature_id] == threshold)
    X_1 = X[true_indices]
    y_1 = y[true_indices]
    X_2 = X[~true_indices]
    y_2 = y[~true_indices]

    return X_1, y_1, X_2, y_2

  def majority_vote(self, y):
    a = y.flatten()
    labels = list(set(a))
    freq = np.array([(a==i).sum() for i in labels])
    label = labels[np.argmax(freq)]
    return label

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    self.root = self.grow_tree(X, y)

  def grow_tree(self, X, y, current_depth=0):
    # here we recursively grow the tree starting from the root
    # the depth of the tree is recorded so that we can later stop if we
    # reach the max_depth

    largest_purity_gain = 0 # initial small value for purity gain
    nr_samples, nr_features = np.shape(X)

    # checking if we have reached the pre-specified limits
    if nr_samples >= self.min_samples_split and current_depth <= self.max_depth:

        # go over the features to select the one that gives more purity
        for feature_id in range(nr_features):                      
          
          unique_values = np.unique(X[:, feature_id])
          
          # we iterate through all unique values of feature column and
          # calculate the impurity
          for threshold in unique_values: 

            # Divide X and y according to the condition 
            # if the feature value of X at index feature_id
            # meets the threshold
            X1, y1, X2, y2 = self.divide_on_feature(X, y, feature_id, threshold)                        

            # checking if we have samples in each subtree
            if len(X1) > 0 and len(X2) > 0:               
                # calculate purity gain for the split
                purity_gain = self.calculate_purity_gain(y, y1, y2)

                # If this threshold resulted in a higher purity gain than 
                # previously thresholds store the threshold value and the 
                # corresponding feature index
                if purity_gain > largest_purity_gain:
                  largest_purity_gain = purity_gain
                  best_feature_id = feature_id
                  best_threshold = threshold
                  best_X1 = X1 # X of right subtree (true)
                  best_y1 = y1 # y of right subtree (true)
                  best_X2 = X2 # X of left subtree (true)
                  best_y2 = y2 # y of left subtree (true)

    # if the resulting purity gain is good enough according our 
    # pre-specified amount, then we continue growing subtrees using the
    # splitted dataset, we also increase the current_depth as 
    # we go down the tree
    if largest_purity_gain > self.min_impurity:

      true_branch = self.grow_tree(best_X1,
                                    best_y1,
                                    current_depth + 1)
      
      false_branch = self.grow_tree(best_X2,
                                    best_y2,
                                    current_depth + 1)

      return DecisionNode(feature_id=best_feature_id,
                          threshold=best_threshold,
                          true_branch=true_branch,
                          false_branch=false_branch)

    # If none of the above conditions are met, then we have reached the 
    # leaf of the tree  and we need to store the label
    leaf_value = np.average(y)

    return DecisionNode(value=leaf_value)


  def predict_value(self, x, tree=None):
    # this is a helper function for the predict method
    # it recursively goes down the tree
    # x is one instance (row) of our test dataset

    # when we don't specify the tree, we start from the root
    if tree is None:
        tree = self.root

    # if we have reached the leaf, then we just take the value of the leaf
    # as prediction
    if tree.value is not None:
        return tree.value

    # we take the feature of the current node that we are on
    # to test whether our instance satisfies the condition
    feature_value = x[tree.feature_id]

    # determine if we will follow left (false) or right (true) branch
    # down the tree
    branch = tree.false_branch
    if isinstance(feature_value, int) or isinstance(feature_value, float):
        if feature_value >= tree.threshold:
            branch = tree.true_branch
    elif feature_value == tree.threshold:
        branch = tree.true_branch

    # continue going down the tree recursively through the chosen subtree
    # this function will finish when we reach the leaves 
    return self.predict_value(x, branch)

  def predict(self, X):
    # Classify samples one by one and return the set of labels 
    X = np.array(X)
    y_pred = [self.predict_value(instance) for instance in X]
    return y_pred

class SVR:
  """
  Hard (C=0) and Soft (C>0) Margin Support Vector Machine classifier 
  with kernels
  """
  def __init__(self, epsilon=0.1, C=1, kernel_name='linear', power=2, gamma=None, coef=2):  
    self.C = C
    self.power = power # degree of the polynomial kernel (d in the slides) 
    self.gamma = gamma # Kernel coefficient for "rbf" and "poly"
    self.coef = coef # coefficent of the polynomial kernel (r in the slides)
    self.kernel_name = kernel_name  # implement for 'linear', 'poly' and 'rbf'
    self.kernel = None
    self.alphas = None
    self.other_alphas = None
    self.support_vectors = None
    self.support_vector_labels = None
    self.t = None
    self.epsilon=epsilon

  def get_kernel(self, kernel_name):

    def linear(x1, x2): return np.dot(x1, x2)
    def polynomial(x1, x2): return (self.gamma * np.dot(x1, x2) + self.coef)**self.power
    def rbf(x1, x2): return np.exp(-1 * self.gamma * np.dot(x1-x2, x1-x2))
    
    kernel_functions = {'linear': linear,
                        'poly': polynomial,
                        'rbf': rbf}

    return kernel_functions[kernel_name]

  def fit(self, X, y):
  
    nr_samples, nr_features = np.shape(X)
    # Setting a default value for gamma
    if not self.gamma:
      self.gamma = 1 / nr_features
    
    # Set the kernel function
    self.kernel = self.get_kernel(self.kernel_name)
    # Construct the kernel matrix
    X_np = np.array(X)
    kernel_matrix = np.zeros((nr_samples, nr_samples))  
    for i in range(nr_samples):
        for j in range(nr_samples):
            kernel_matrix[i, j] = self.kernel(X_np[i], X_np[j])
    e = np.ones(nr_samples)
    # Define the quadratic optimization problem
    M_n = np.concatenate((np.identity(nr_samples), -1*np.identity(nr_samples)), axis=1)
    M_p = np.concatenate((np.identity(nr_samples), np.identity(nr_samples)), axis=1)
    get_norm = np.concatenate((np.identity(nr_samples), 0*np.identity(nr_samples)), axis=1)
    get_other = np.concatenate((0*np.identity(nr_samples), np.identity(nr_samples)), axis=1)
    
    P = cvxopt.matrix(M_n.T@kernel_matrix@M_n, tc='d')
    q = cvxopt.matrix(self.epsilon*M_p.T@e - M_n.T@y)
    A = cvxopt.matrix(e.T@M_n, (1, 2*nr_samples), tc='d')
    b = cvxopt.matrix(0, tc='d')

    if not self.C:
        G = cvxopt.matrix(np.identity(nr_samples) * -1)
        h = cvxopt.matrix(np.zeros(nr_samples))
    else:
        G_max = -1*np.identity(2*nr_samples)
        G_min = np.identity(2*nr_samples)
        G = cvxopt.matrix(np.vstack((G_max, G_min)))
        h_max = cvxopt.matrix(np.zeros(2*nr_samples))
        h_min = cvxopt.matrix(np.ones(2*nr_samples) * self.C)
        h = cvxopt.matrix(np.vstack((h_max, h_min)))

    # Solve the quadratic optimization problem using cvxopt
    minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    alphas = np.ravel(minimization['x'])
    alphas = M_n@alphas

    # first get indexes of non-zero lagr. multipiers
    idx = alphas > 1e-7

    # get the corresponding lagr. multipliers (non-zero alphas)
    self.alphas = alphas[idx]

    # get the support vectors
    self.support_vectors = np.array(X)[idx]
    
    # get the corresponding labels
    self.support_vector_labels = np.array(y)[idx]
    
    # Calculate intercept (t) with first support vector
    self.t = np.array([self.alphas[i]*self.support_vector_labels[i]*self.kernel(self.support_vectors[i], self.support_vectors[0]) for i in range(len(self.alphas))]).sum(axis=0)  - self.support_vector_labels[0]
          
  def predict(self, X):
    y_pred = []
    for vec in np.array(X):
        y_pred.append(np.array([self.alphas[i]*self.support_vector_labels[i]*self.kernel(self.support_vectors[i], vec) for i in range(len(self.alphas))]).sum(axis=0)-self.t)
    return np.array(y_pred)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
   def __init__(self, learning_rate=1e-3, nr_iterations=10, batch_size=64):
    self.alearning_rate = learning_rate
    self.iterations = nr_iterations
    self.batch_size = 64
    self.weights = np.random.normal(0, 1, a.shape)
    
   def fit(self, X, y):
    converged = False

    # we can store the loss values to see how fast the algorithm converges
    self.loss = []

    # a counter of algorithm steps
    i = 0 

    while (not converged):
      i += 1

      # calculate the predictions in case of the weights in this stage
      y_pred = [sigmoid(X[i]@self.weights) for i in range(X.shape[0])]

      # calculate the mean squared error (loss) for the predictions
      # obtained above
      self.loss.append(y*np.log(y_pred) + (1-y)*np.log(y_pred))

      # calculate the gradient of the objective function with respect to w
      # for the second component \sum|w_i| use np.sign(w_i) as it's derivative
      grad = np.sum([X[i]*(y[i]-y_pred[i]) for i in range(X.shape[0])])
      new_weights = self.weights - self.learning_rate * grad

      converged = (i>self.iterations)
      self.weights = new_weights
   
   def predict(self, X):
    y_pred = [sigmoid(X[i]@self.weights) for i in range(X.shape[0])]
    return y_pred

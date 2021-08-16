import numpy as np
import random 


def euclid_dist(p1, p2):
  return (np.sum((p1-p2)*(p1-p2)))**0.5


class KMeans:
  def __init__(self, k=2, max_iterations=500, tol=0.5):
    # number of clusters
    self.k = k
    # maximum number of iterations to perform
    # for updating the centroids
    self.max_iterations = max_iterations
    # tolerance level for centroid change after each iteration
    self.tol = tol
    # we will store the computed centroids 
    self.centroids = None

  def init_centroids(self, X):
    X = np.array(X)
    
    a = np.random.choice(np.arange(X.shape[0]), self.k)
    
    centroids = X[a]
    
    return centroids

  def closest_centroid(self, X):
    
    dist_matrix = np.array([[np.linalg.norm(x-self.centroids[j]) for j in range(self.k)] for x in X])
    
    nearest_cluster = np.argmin(dist_matrix, axis=1)

    return nearest_cluster

  def update_centroids(self, X, label_ids):

    new_centroids = [np.average(X[label_ids==j], axis=0) for j in range(self.k)]
    
    new_centroids = np.array(new_centroids)
    return new_centroids

  def fit(self, X):
    # this is the main method of this class
    X = np.array(X)

    # we start by random centroids from our data
    self.centroids = self.init_centroids(X)

    not_converged = True
    i = 1 # keeping track of the iterations

    while not_converged and (i < self.max_iterations):
      current_labels = self.closest_centroid(X)
      new_centroids = self.update_centroids(X, current_labels)

      # count the norm between new_centroids and self.centroids
      # to measure the amount of change between 
      # old cetroids and updated centroids
      norm = np.linalg.norm(new_centroids-self.centroids)
      not_converged = norm > self.tol
      self.centroids = new_centroids
      i += 1
    self.labels = current_labels
    print(f'Converged in {i} steps')

  def predict(self, X):

    X = np.array(X)

    return self.closest_centroid(X)

class Cluster:
  def __init__(self, points):
    self.points_list = np.array(points)[np.newaxis, :]
    
  def merge(self, cluster):
        
    new_points = np.append(self.points_list, cluster.points_list, axis=0)
        
    self.points_list = new_points
        
  def distance(self, cluster, diss_func, linkage):
        
    min = 1000000
    max = 0

    for point1 in self.points_list:
      for point2 in cluster.points_list:
        if len(point1)*len(point2) == 0:
          return 1000000
        dist = diss_func(point1, point2)
        if dist > max:
          max = dist
            
        if dist < min:
          min = dist
            
    if linkage == 'single':
      return min
    
    else:
      return max

  def display(self):
    print("Beginning of cluster")
    for point in self.points_list:
      print(point)
    print("End of cluster")

  def is_empty(self):
    return self.points_list.size == 0

class HierarchicalClustering:
 

  def __init__(self, nr_clusters, diss_func=euclid_dist, linkage='single', distance_threshold=None):
    
    self.nr_clusters = nr_clusters
    
    self.diss_func = diss_func
    
    self.linkage = linkage
    
    self.thresh = distance_threshold
    
  
  def fit(self, X):
    X = np.array(X)
    
    real_nr = X.shape[0]
    
    clusters = [Cluster(x) for x in X]
    
    while real_nr > self.nr_clusters:

      dist_matrix = []

      for cluster1 in clusters:

        for cluster2 in clusters:

          dist_matrix.append(cluster1.distance(cluster2, self.diss_func, self.linkage))

      dist_matrix = np.array(dist_matrix)
      dist_matrix = dist_matrix.reshape(real_nr, -1)
      dist_matrix += (np.eye(real_nr)*np.max(dist_matrix)).astype('int')

      cluster_coord = np.argwhere(dist_matrix == np.min(dist_matrix))[0]

      closest1 = clusters[cluster_coord[0]]
      closest2 = clusters[cluster_coord[1]]

      closest1.merge(closest2)
      clusters.remove(closest2)

      real_nr -= 1   

      self.clusters = clusters
    
    for cluster in clusters:
      cluster.display()

  def predict(self, X):
    X = [Cluster(x) for x in np.array(X)]
    dists = []
    labels = []

    for cluster in self.clusters:
      dists = []
      for x in X:
        dists.append(cluster.distance(x, diss_func, linkage))
      dists = np.array(dists)
      labels.append(np.argmin(dists))

    return np.array(labels)
                             
class DBSCAN:
  def __init__(self, diss_func=euclid_dist, epsilon=0.5, min_points=5):
    self.diss_func = diss_func
    
    self.e = epsilon
    
    self.min_points = min_points
    
    self.clusters = None

  def fit(self, X):
    X = np.array(X)
    clusters = [Cluster(x) for x in X]

    for cluster1 in clusters:

      for cluster2 in clusters:

        if cluster1 != cluster2:

          if cluster1.distance(cluster2, self.diss_func, linkage='single') <= self.e:

            cluster1.merge(cluster2)

            cluster2.points_list = np.array([[]])

    self.clusters = []

    for cluster in clusters:
      if not cluster.is_empty():
        self.clusters.append(cluster)

    for cluster in self.clusters:
      cluster.display()
    
    
  def predict(self, X):
    X = [Cluster(x) for x in np.array(X)]
    labels = []

    for x in X:

      found = False

      for cluster in self.clusters:

        if x.distance(cluster) <= epsilon:
          if len(cluster.points_list)<self.min_points:
            labels.append(-1)
            found = True

          else:
            labels.append(self.clusters.index(cluster))
            found = True
          break

      if found == False:
        labels.append(-1)

    return np.array(labels)


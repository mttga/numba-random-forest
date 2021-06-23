import numpy as np
from cart_utils import Node, get_tree, node_predict, max_depth, number_leaves, count_attributes

class RandomFeatures():
    """
    An object of this class can sample a subset of a original
    feature space given one of the followings methods (1, 3, log, sqrt)
    """  
    
    methods = {'single', 'triple', 'log', 'sqrt'}
    
    def __init__(self, n_features=0, method='sqrt'):
        
        self.n_features = n_features
        if method  == 'single':
            self.F = 1
        elif method == 'triple':
            self.F = 3
        elif method == 'log':
            self.F = int(np.log2(n_features)+1)
        elif method == 'sqrt':
            self.F = int(np.sqrt(n_features))
        else:
            raise ValueError('The admitted methods are: single, triple, log, sqrt')
    
    def sample(self):
        return sorted(np.random.choice(self.n_features, self.F, replace=False))

    
class RandomCart():
    """
    Basic Cart algorithm that can use a random subset of the original feature space
    at each node split
    """
    
    def __init__(self, discrete_thr=10):
        self.tree = Node()
        self.features = []
        self.discrete = []
        self.n_attributes = 0
        self.random_features = None
        self.discrete_thr = discrete_thr # threshold to consider a feature discrete

    def fit(self, X, y, random_features=None):
        self.n_attributes = X.shape[1]
        self.discrete = np.where([np.unique(X[:,a]).size <= self.discrete_thr for a in range(X.shape[1])], 1, 0)
        self.random_features = random_features
        get_tree(X, y, self.tree, self.discrete, self.random_features)
        
    def predict(self, X):
        return [node_predict(self.tree, x) for x in X]
        
    def depth(self):
        return max_depth(self.tree)
    
    def n_leaves(self):
        return number_leaves(self.tree)
    
    def attributes_frequency(self):

        self.att_counts = np.zeros(self.n_attributes, dtype=np.int)
        count_attributes(self.tree, self.att_counts)

        return self.att_counts

class RandomForest():
    
    def __init__(self, n_trees=100, random_method='sqrt', max_samples=0.66):
        
        if random_method not in RandomFeatures.methods: raise ValueError('The admitted F are: {}'.format(RandomFeatures.methods))
        self.trees = []
        self.n_trees = n_trees
        self.n_attributes = 0
        self.random_method = random_method
        self.random_features = None
        self.max_samples = max_samples
        
    def fit(self, X, y):
        
        self.n_attributes = X.shape[1]
        self.n_instances = X.shape[0]
        self.random_features = RandomFeatures(self.n_attributes, self.random_method)
        self.bootsrap_size = int(self.n_instances*self.max_samples)
        
        # Fit a cart tree with a subsample of the features
        for _ in range(self.n_trees):
            indexes = np.random.choice(self.n_instances, size=self.bootsrap_size, replace=False) # Bootstrap
            X_bootstrap, y_bootsrap = X[indexes], y[indexes]
            tree = RandomCart()
            tree.fit(X_bootstrap, y_bootsrap, self.random_features)
            self.trees.append(tree)
     
    def predict(self, X):
        
        # Predict all the values of X with all the trees and save the results 
        # in a matrix with n_trees*n_predictions
        pred_matrix = np.zeros((self.n_trees, X.shape[0]), dtype=np.int)
        for i in range(self.n_trees):
            pred_matrix[i] = self.trees[i].predict(X)
        # Take the most frequent label for each instance
        labels = []
        for i in range(pred_matrix.shape[1]):
            labels.append(np.bincount(pred_matrix[:,i]).argmax())
        return labels
    
    def features_importance(self):
        
        # Retrieve a feature matrix with the frequency of all the 
        # attributes for each tree
        features_matrix = np.zeros((self.n_trees, self.n_attributes))
        for i in range(self.n_trees):
            features_matrix[i] = self.trees[i].attributes_frequency()
        
        # The importance of a feature the sum of its frequency for all
        # the trees normalized by the total number of features splits
        return np.sum(features_matrix, axis=0)/np.sum(features_matrix)
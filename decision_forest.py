import numpy as np
from cart_utils import Node, get_tree, node_predict, max_depth, number_leaves, count_attributes

rng = np.random.default_rng()

"""Functions that randomly return a subset of M features"""
def div2_F(M):
    F = int(M/2)
    return sorted(np.random.choice(M, F, replace=False))

def div4_F(M):
    F = int(M/4)
    return sorted(np.random.choice(M, F, replace=False))
    
def div4_3F(M):
    F = int(3*M/4)
    return sorted(np.random.choice(M, F, replace=False))

def rand_F(M):
    F = rng.integers(low=1, high=M, endpoint=True)
    return sorted(np.random.choice(M, F, replace=False))

random_feat_dict = {'div2_F':div2_F, 'div4_F':div4_F, 'div4_3F':div4_3F, 'rand_F':rand_F}


class Cart():
    """Basic Cartt algorithm that can grow a tree with a subset of features"""
    def __init__(self, discrete_thr=10):
        self.tree = Node()
        self.features = [] # array of the features considered when growing the tree
        self.discrete = [] # array that takes trace of the discrete and continous features
        self.n_attributes = 0
        self.discrete_thr = discrete_thr # threshold to consider a feature discrete (considering the number of instances of the attribute)

    def fit(self, X, y, features=[]):
        self.n_attributes = X.shape[1]
        self.discrete = np.where([np.unique(X[:,a]).size <= self.discrete_thr for a in range(X.shape[1])], 1, 0)
        
        # If a subset of features is given, cart operates only on that
        if features:
            self.features = features
            X = X[:,self.features]
            self.discrete = self.discrete[self.features]
        get_tree(X, y, self.tree, self.discrete)
        
    def predict(self, X):
        if self.features:
            X = X[:,self.features]
        return [node_predict(self.tree, x) for x in X]
        
    def depth(self):
        return max_depth(self.tree)
    
    def n_leaves(self):
        return number_leaves(self.tree)
    
    def attributes_frequency(self):

        if self.features:
            # If the training was operated in a subset of the training set, 
            # retrieve the counties for only the features used in the training
            feat_counts = np.zeros(len(self.features), dtype=np.int)
            count_attributes(self.tree, feat_counts)
            # And then concatenate with all the attributes
            self.att_counts = np.zeros(self.n_attributes, dtype=np.int)
            self.att_counts[self.features] = feat_counts
        else: 
            self.att_counts = np.zeros(self.n_attributes, dtype=np.int)
            count_attributes(self.tree, self.att_counts)

        return self.att_counts


class DecisionForest():
    def __init__(self, n_trees=100, F='rand_F'):
        if F not in random_feat_dict.keys(): raise ValueError('The admitted F are: {}'.format(random_feat_dict.keys()))     
        self.trees = []
        self.n_trees = n_trees
        self.n_attributes = 0
        self.random_features = random_feat_dict[F]
        
    def fit(self, X, y):
        self.n_attributes = X.shape[1]
        # Fit a cart tree with a subsample of the features
        for _ in range(self.n_trees):
            features = self.random_features(self.n_attributes)
            tree = Cart()
            tree.fit(X, y, features)
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
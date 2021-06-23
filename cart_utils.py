import numpy as np
import numba as nb
from numba import jit
from numba.experimental import jitclass
from collections import OrderedDict

node_type = nb.deferred_type()
spec = OrderedDict()
spec['att'] = nb.int32
spec['val'] = nb.float64[:]
spec['discrete'] = nb.boolean
spec['depth'] = nb.int32
spec['child1'] = nb.optional(node_type)
spec['child2'] = nb.optional(node_type)
spec['is_leaf'] = nb.boolean
spec['cl'] = nb.int32

@jitclass(spec)
class Node:
    def __init__(self):
        self.att = -1
        self.val = np.empty(1)
        self.discrete = False
        self.depth = 0
        self.child1 = None
        self.child2 = None
        self.is_leaf = False
        self.cl = -1

node_type.define(Node.class_type.instance_type)


@jit(nopython=True)
def get_gini(labels):
    classes = np.unique(labels)
    total_instances = labels.size
    impurity = 0
    for c in classes:
        impurity += (np.count_nonzero(labels==c)/total_instances)**2
    return 1 - impurity


@jit(nopython=True)
def get_gini_partition(X1, X2, y1, y2):
    size1 = X1.size
    size2 = X2.size
    total_size = size1 + size2
    return (size1/total_size)*get_gini(y1) + (size2/total_size)*get_gini(y2)


@jit(nopython=True)
def get_indexes_in_arr(X, arr):
    """
    Functions that return the indexes 
    of the values an array X that are present 
    in another array arr
    """
    indexes = np.full(X.size, False)
    for x in range(X.size):
        for a in range(arr.size):
            if X[x] == arr[a]:
                indexes[x] = True
                break
    return indexes

@jit(nopython=True)
def combinations(arr, comb_size):
    """
    Functions that return all the combinations
    without permutation of an array given a comb_size
    """
    n = arr.size
    indices = np.arange(comb_size)
    empty = not(n and (0 < comb_size <= n))

    if not empty:
        yield arr[indices]

    while not empty:
        i = comb_size - 1
        while i >= 0 and indices[i] == i + n - comb_size:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i+1, comb_size):
                indices[j] = indices[j-1] + 1

            yield arr[indices]


@jit(nopython=True)
def best_partition(X, y, discrete):
    """
    Function that finds the partition of X that maximizes the reduction of impurity
    Args:
        - X (numpy 2-dimensional array): the train set or its subset
        - y (numpy 1-dimensional array): the labels of the train set or a subset of them
        - discrete (boolean array): 1 -> discrete feature, 0 -> continuous feature
    Returns:
        - best_attr (integer): the index of the attribute that gives the best split
        - best_value (numpy 1-dimensional array): the values that give the best split
        – best_indexes (numpy 1-dimensional boolean array): True -> the value belongs to 
        child1, False –> the value belong to child2
    """

    best_gini_partition = np.inf
    best_attr = 0
    best_value = np.empty(1)
    best_indexes = np.full(y.size, False)

    for a in range(X.shape[1]):
        unique_values = np.unique(X[:, a])
        
        # If the attribute is discrete we consider the attribute discrete
        # we take in consideration all the possible combinations of 
        # the unique values
        if discrete[a] == 1:
            for comb_size in range(1, unique_values.size):
                for c in combinations(unique_values,comb_size):
                    indexes = get_indexes_in_arr(X[:,a],c)
                    X1, y1 = X[indexes], y[indexes]
                    X2, y2 = X[~indexes], y[~indexes]
                    gini_partition_temp = get_gini_partition(X1, X2, y1, y2)

                    if gini_partition_temp < best_gini_partition:
                        best_gini_partition = gini_partition_temp
                        best_attr  = a
                        best_value = c.astype(np.float64)
                        best_indexes = indexes
        
        # If there the attribute is continuous we need to menage the mean values
        else:
            mean_values = []
            for u in range(0, unique_values.size-1):
                mean_values.append((unique_values[u] + unique_values[u+1])/2)
            for m in mean_values: 
                indexes = X[:, a] <= m
                X1, y1 = X[indexes], y[indexes]
                X2, y2 = X[~indexes], y[~indexes]
                gini_partition_temp = get_gini_partition(X1, X2, y1, y2)

                if gini_partition_temp < best_gini_partition:
                    best_gini_partition = gini_partition_temp
                    best_attr  = a
                    best_value = np.array([m])
                    best_indexes = indexes

    return best_attr, best_value, best_indexes


def get_tree(X, y, node, discrete, random_features=None, depth=1):
    """
    Functions that expand recursively a tree
    using the best_partition function. It can use a 
    random_features generator in order to sample
    randomly the features used to split the tree at each node
    Args:
        - X (numpy 2-dimensional array): train set
        - y (numpy 1-dimensional array): labels of the train set
        - discrete (boolean vector): 1 for discrete attributes, 0 for continous
        - random_features (RandomFeatures): generator of features indexes
        - depth (int): depth of the recursions (of the growing tree)
    """

    # Base case: all the labels are the same: in this case
    # we found a perfect slipt and we don't need to split the 
    # tree anymore
    if np.unique(y).size == 1:
        node.cl = np.unique(y)[0]
        node.is_leaf = True  
        node.depth = depth
    else:
        
        if random_features:
            # In this case we need to use the random_features generator
            # to sample a subset of features and obtain the best_partition
            # in this subsample. The attribute returned must be considered
            # as the index of the chosen attribute in the features subset
            features = random_features.sample()
            att_index, val, indexes = best_partition(X[:,features], y, discrete[features])
            att = features[att_index]
        else:
            att, val, indexes = best_partition(X, y, discrete)
        
        # Recursive step: 
        if indexes.any():
            X1, y1 = X[indexes], y[indexes]
            X2, y2 = X[~indexes], y[~indexes]
            node.att = att
            node.val = val
            node.discrete = discrete[att]
            node.child1 = Node()
            node.child2 = Node()
            get_tree(X1, y1, node.child1, discrete, random_features, depth+1)
            get_tree(X2, y2, node.child2, discrete, random_features, depth+1)
        
        # Could be the case that the labels are not unique but it's
        # not possible to find a good split of the tree (for example, 
        # if we have many instances that are equal to each other but 
        # have different classes): in this case a good euristich is to
        # stop to split the tree and take the most frequent label
        else:
            node.cl = np.bincount(y).argmax()
            node.is_leaf=True
            node.depth = depth
            
@jit(nopython=True)
def number_leaves(node, count=0):
    if node.is_leaf:
        return count + 1
    else:
        return count + number_leaves(node.child1, count) + number_leaves(node.child2, count)

    
@jit(nopython=True)
def max_depth(node):
    if node.is_leaf:
        return node.depth
    else:
        return max(max_depth(node.child1), max_depth(node.child2))

    
@jit(nopython=True)
def node_predict(node, x):
    """
    Function that permits to predict the label of an instance
    given the root node of a tree
    """
    
    # Base case: the node is a leaf -> return the class
    if node.is_leaf:
        return node.cl
    
    # In the other cases we need to follow the correct tree branches
    x_val = x[node.att]
    
    # Recrsive case one: the current attribute is discrete -> check 
    # if x is in the set of values of the node
    if node.discrete:          
        is_in_val = False
        for v in node.val:
            if x_val == v:
                is_in_val = True
                break           
        if is_in_val:
            return node_predict(node.child1, x)
        else:
            return node_predict(node.child2, x)
            
    
    # Recursive cases two: the attribute is not discrete -> check
    # if x is minor of the value of the node
    if not node.discrete:
        if x_val <= node.val[0]:
            return node_predict(node.child1, x)
        else:
            return node_predict(node.child2, x)

@jit(nopython=True)
def predict_node_many(node, X):
    labels = []
    for x in X:
        labels.append(node_predict2(node, x))
    return labels

@jit(nopython=True)
def count_attributes(node, att_counts):
    if node.att != -1:
        att_counts[node.att] += 1
    if not node.is_leaf:
        count_attributes(node.child1, att_counts)
        count_attributes(node.child2, att_counts)
        

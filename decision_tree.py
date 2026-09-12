import numpy as np

class Node():
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
  
class BaseDecisionTree():
    def __init__(self, max_depth = 12):
        self.root = None
        self.max_depth = max_depth
    
    def _calculate_impurity(self, y):
        raise NotImplementedError
        
    def _calculate_leaf_value(self, y):
        raise NotImplementedError
        
    def _information_gain(self, y, y_left, y_right):
        """Retourne le gain d'information dans les noeuds fils par rapport au noeud parent"""
        W_left = len(y_left)/len(y)
        W_right = len(y_right)/len(y)
        return self._calculate_impurity(y) - W_left*self._calculate_impurity(y_left) - W_right*self._calculate_impurity(y_right)
  
        
    def _split(self, X_column, threshold):
        """Renvoie les indices des données qui vont à gauche ou à droite de l'arbre, par rapport au seuil"""
        left = np.where(X_column <= threshold)[0]
        right = np.where(X_column > threshold)[0]
        return left, right
    
    def _get_best_split(self, X, y):
        """Renvoie la meilleure division du noeud parent en comparant les gains d'information"""
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            X_column = X[:,feature_idx]
            candidats = np.unique(X_column)
            for threshold in candidats:
                left, right = self._split(X_column, threshold)
                
                if len(left) == 0 or len(right) == 0:
                    continue
                    
                y_left, y_right = y[left], y[right]
                
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    
        return best_gain, best_feature_idx, best_threshold
    
    def _build_tree(self, X, y, current_depth=0):
        """Construit récursivement l'arbre de décision"""
        # Condition d'arrêt
        if len(set(y)) <= 1 or current_depth == self.max_depth:
            #labels, counts = np.unique(y, return_counts=True)
            node = Node()
            node.value = self._calculate_leaf_value(y)
            return node
        
        gain, feature_idx, threshold = self._get_best_split(X, y)
        
        if gain <= 0:
            #
            node = Node()
            node.value = self._calculate_leaf_value(y)
            return node
        
        X_column = X[:, feature_idx]
        left, right = self._split(X_column, threshold)
        X_left, y_left = X[left], y[left]
        X_right, y_right = X[right], y[right]
        
        node = Node()
        node.feature = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(X_left, y_left, current_depth + 1)
        node.right = self._build_tree(X_right, y_right, current_depth + 1)
        
        return node
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
        
        
class DecisionTreeClassifier(BaseDecisionTree):    
    def _calculate_impurity(self, y):
        """Retourne le score d'impureté de Gini"""
        if len(y) == 0:
            return 0
        else:
            _, counts = np.unique(y, return_counts=True)
            N = len(y)
            return 1 - np.sum((counts/N)**2)
        
    def _calculate_leaf_value(self, y):
        """La classe majoritaire pour la classification"""
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

class DecisionTreeRegressor(BaseDecisionTree):          
    def _calculate_impurity(self, y):
        """Variance (MSE) pour la régression"""
        if len(y) == 0:
            return 0
        return np.var(y)
        
    def _calculate_leaf_value(self, y):
        """La moyenne des cibles pour la régression"""
        return np.mean(y)

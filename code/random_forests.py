import numpy as np
from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

class BaseRandomForest():
    def __init__(self, n_trees = 10, max_depth = 12):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """Créer un échantillon aléatoire avec remise"""
        n_samples = X.shape[0]
        rand_indices = np.random.choice(n_samples, size = n_samples, replace = True)
        return X[rand_indices], y[rand_indices]
            
    def _make_tree(self):
        raise NotImplementedError

    def _aggregate_predictions(self, votes_per_sample):
        raise NotImplementedError
    
    def fit(self, X, y):
        """Créer et entraîner n_trees arbres"""
        for i in range(self.n_trees):
            # On crée UN échantillon Bootstrap pour cet arbre individuel
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = self._make_tree()
            # On entraîne cet arbre sur cet échantillon et on l'ajoute à la forêt
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        # Recolter les prédictions de chaque arbre
        tree_predictions = [tree.predict(X) for tree in self.trees]
            
        # Transposition pour avoir une matrice (n_samples, n_trees)
        votes_per_sample = np.array(tree_predictions).T
        
        # Prédiction finale (vote majoritaire)
        return self._aggregate_predictions(votes_per_sample)
        

class RandomForestClassifier(BaseRandomForest):
    def _make_tree(self):
        return DecisionTreeClassifier(max_depth=self.max_depth)
    
    def _aggregate_predictions(self, votes_per_sample):
        # Pour la classification, vote majoritaire sur chaque ligne
        y_pred = []
        for votes in votes_per_sample:
            labels, counts = np.unique(votes, return_counts=True)
            majorite = labels[np.argmax(counts)]
            y_pred.append(majorite)
        return np.array(y_pred)
        

class RandomForestRegressor(BaseRandomForest):
    def _make_tree(self):
        return DecisionTreeRegressor(max_depth=self.max_depth)
    
    def _aggregate_predictions(self, votes_per_sample):
        # Pour la régression, on prend simplement la moyenne sur chaque ligne
        return np.mean(votes_per_sample, axis=1)

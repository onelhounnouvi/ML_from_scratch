import numpy as np
from decision_tree import DecisionTreeRegressor

class BaseGradientBoosting():
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, n_classes=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.trees = []
        self.F0 = None
        
    def _compute_initial_prediction(self, y):
        raise NotImplementedError
    
    def _compute_residuals(self, y, F):
        raise NotImplementedError
    
    def _transform_predictions(self, F):
        raise NotImplementedError

    def fit(self, X, y):
        # Initialisation de F0
        self.F0 = self._compute_initial_prediction(y)
        n_samples = X.shape[0]
        
        # -------------------------------------------------------------
        # CAS 1 : Sortie Unidimensionnelle (Régression / Binaire)
        # -------------------------------------------------------------
        if self.n_classes == 1:
            # Initialisation de F(x)
            F = np.full(n_samples, self.F0, dtype=float)
            
            # Boucle pour n_estimators:
            for _ in range(self.n_estimators):
                # Calcul des résidus
                r = self._compute_residuals(y, F)
                
                # Fit de l'arbre de régression sur (X, résidus)
                tree = DecisionTreeRegressor(self.max_depth)
                tree.fit(X, r)
                
                # Mise à jour de F(x)
                F += self.learning_rate * tree.predict(X)
                
                # Stockage de l'arbre
                self.trees.append(tree)
                
        # -------------------------------------------------------------
        # CAS 2 : Sortie Multiclasse (K > 1, matrice (N, K))
        # -------------------------------------------------------------
        else:
            # Initialisation de la matrice F à F0 pour chaque classe (N, K)
            F = np.full((n_samples, self.n_classes), self.F0, dtype=float)
            
            # Boucle pour n_estimators:
            for _ in range(self.n_estimators):
                # Calcul des résidus : matrice N*K
                R = self._compute_residuals(y, F)
                trees_in_step = []

                # Entraînement de K arbres (un par classe)
                for k in range(self.n_classes):
                    tree = DecisionTreeRegressor(self.max_depth)
                    tree.fit(X, R[:,k])
                    
                    # Mise à jour de F(x)
                    F[:,k] += self.learning_rate * tree.predict(X)
                    trees_in_step.append(tree)
                    
                # Stockage des arbres
                self.trees.append(trees_in_step)

    def predict(self, X):
        n_samples = X.shape[0]
        # -------------------------------------------------------------
        # CAS 1 : Sortie Unidimensionnelle (Régression / Binaire)
        # -------------------------------------------------------------
        if self.n_classes == 1:
            # 1. Initialiser F avec n_samples fois F0
            F = np.full(n_samples, self.F0, dtype=float)
            
            # 2. Sommer les prédictions pondérées de chaque arbre
            for tree in self.trees:
                F += self.learning_rate * tree.predict(X)

        # -------------------------------------------------------------
        # CAS 2 : Sortie Multiclasse (K > 1, matrice (N, K))
        # -------------------------------------------------------------
        else:
            # On part d'une matrice F0 de taille (N, K)
            F = np.full((n_samples, self.n_classes), self.F0, dtype=float)
            
            # On parcourt chaque itération (qui contient K arbres)
            for trees_in_step in self.trees:
                for k in range(self.n_classes):
                    F[:, k] += self.learning_rate * trees_in_step[k].predict(X)
                    
        # 3. Transformation finale (identité, sigmoïde ou softmax)
        return self._transform_predictions(F)
        
        
class GradientBoostingClassifier(BaseGradientBoosting):
    @staticmethod
    def sigmoid(x):
        """Fonction d'activation sigmoïde"""
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def softmax(X):
        """Fonction d'activation softmax stable numériquement"""
        # Protection contre le débordement (Overflow)
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)
        
    def _compute_initial_prediction(self, y):
        if self.n_classes == 1:
            p = np.mean(y)
            return np.log(p / (1 - p))
        else:
            return 0.0

    def _compute_residuals(self, y, F):
        if self.n_classes == 1:
            return y - self.sigmoid(F)
        else:
            probabilities = self.softmax(F)
            # Encodage One-Hot de y
            y_one_hot = np.eye(self.n_classes)[y]
            return y_one_hot - probabilities

    def _transform_predictions(self, F):
        """Conversion des log-odds en classes"""
        if self.n_classes == 1:
            probabilities = self.sigmoid(F)
            return (probabilities >= 0.5).astype(int)
        else:
            probabilities = self.softmax(F)
            return np.argmax(probabilities, axis=1)


class GradientBoostingRegressor(BaseGradientBoosting):     
    def _compute_initial_prediction(self, y):
        """Moyenne des valeurs cibles"""
        return np.mean(y)

    def _compute_residuals(self, y, F):
        """Résidus MSE simples"""
        return y-F

    def _transform_predictions(self, F):
        """Pas de transformation pour la régression"""
        return F

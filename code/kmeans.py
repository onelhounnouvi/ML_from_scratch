import numpy as np

class KMeans():
    def __init__(self, n_clusters=5, max_iter=100, tol=1e-4, init="kmeans++"):
        """self.init in {"random", "kmeans++"}"""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.tol = tol
        self.init = init
        self.wcss_history_ = []
        self.inertia_ = None
        
    def kmeans_pp(self, X):
        """Initialisation des centroïdes avec Kmeans++"""
        # Initialize le premier centroïde de manière aléatoire
        firstIndex = np.random.choice(X.shape[0])
        centroids = [X[firstIndex]]
        
        # Tableau pour stocker la distance minimale au carré de chaque point 
        # vers le centroïde le plus proche déjà sélectionné (améliorer la complexité)
        min_sq_distances = np.full(X.shape[0], np.inf)
        
        # Choisir les k-1 centroides restants
        while len(centroids) < self.n_clusters:
            # Calcule le carré de la distance au dernier centroide ajouté le plus proche
            sq_distances = np.sum((X - centroids[-1]) ** 2, axis=1)
            min_sq_distances = np.minimum(min_sq_distances, sq_distances)
                 
            # Choisis le prochain centroide avec une proba proportionnelle à D(x)^2
            probas = min_sq_distances / np.sum(min_sq_distances)
            next_idx = np.random.choice(X.shape[0], p=probas)
            centroids.append(X[next_idx])
        
        return np.array(centroids)
        
    def _inertia(self, X, labels):
        """J(c, mu) avec les centroïdes courants"""
        return np.sum((X - self.centroids[labels]) ** 2)
        
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.wcss_history_ = []
        
        # Initialisation aléatoire ou kmeans++ des centroïdes
        if self.init == "random":
            idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[idx]
        else:
            self.centroids = self.kmeans_pp(X)
        
        # On alterne les 2 étapes
        for _ in range(self.max_iter):
            # Assiggner, à chaque point, le centroide le plus proche
            labels = self.predict(X)
            self.wcss_history_.append(self._inertia(X, labels))
            
            # Mise à jour des centroïdes
            old_centroids = self.centroids.copy()
            
            for j in range(self.n_clusters):
                # Filtrer les points attribués au cluster j
                points_du_cluster = X[labels == j]
                # Recalculer le centroïde (moyenne sur les colonnes, axis=0)
                if len(points_du_cluster) > 0:  # Sécurité si un cluster se retrouve vide
                    self.centroids[j] = np.mean(points_du_cluster, axis=0)
                    
            self.wcss_history_.append(self._inertia(X, labels))
            
            # Condition d'arrêt (convergence)
            if np.allclose(old_centroids, self.centroids, atol=self.tol):
                break
        
        # Coût final
        self.inertia_ = self._inertia(X, self.predict(X))         
                    
    def predict(self, X):
        """On retourne l'indice du centroïde le plus proche pour chaque point"""
        # Vectorisation NumPy pour éviter la boucle Python 
        # X a pour dimension (n_samples, d) -> on le reshape en (n_samples, 1, d)
        # self.centroids a pour dimension (n_clusters, d)
        # Par broadcasting, (n_samples, 1, d) - (1, n_clusters, d) donne (n_samples, n_clusters, d)
        distances = np.sum((X[:, np.newaxis, :] - self.centroids) ** 2, axis=2)
        return np.argmin(distances, axis=1)

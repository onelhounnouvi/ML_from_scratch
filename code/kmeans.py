import numpy as np

class KMeans():
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, init="kmeans++"):
        """self.init in {"random", "kmeans++"}"""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.tol = tol
        self.init = init
        
    def kmeans_pp(self, X):
        """Initialisation des centroïdes avec Kmeans++"""
        # Initialize le premier centroïde de manière aléatoire
        firstIndex = np.random.choice(X.shape[0])
        centroids = [X[firstIndex]]
        
        # Choisir les k-1 centroides restants
        while len(centroids) < self.n_clusters:
            distancesSquared = []
            
            # For each point, compute squared distance to nearest selected centroid
            for x in X:
                distances = np.linalg.norm(x - centroids, axis=1)
                distancesSquared.append(np.min(distances)**2)
                
            # Choose next centroid with probability proportional to D(x)^2
            probas = distancesSquared / np.sum(distancesSquared)
            next_idx = np.random.choice(X.shape[0], p=probas)
            centroids.append(X[next_idx])
        
        return np.array(centroids)
        
    def fit(self, X):
        # Initialisation aléatoire ou kmeans++ des centroïdes
        if self.init == "random":
            idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[idx]
        else:
            self.centroids = self.kmeans_pp(X)
        
        for _ in range(self.max_iter):
            labels = self.predict(X)
            
            # Mise à jour des centroïdes
            old_centroids = self.centroids.copy()
            for j in range(self.n_clusters):
                # Filtrer les points attribués au cluster j
                points_du_cluster = X[labels == j]
                
                # Recalculer le centroïde (moyenne sur les colonnes, axis=0)
                if len(points_du_cluster) > 0:  # Sécurité si un cluster se retrouve vide
                    self.centroids[j] = np.mean(points_du_cluster, axis=0)
            
            # Condition d'arrêt (convergence)
            if np.allclose(old_centroids, self.centroids, atol=self.tol):
                break
                    
    def predict(self, X):
        labels = []
        for x in X:
            distances = np.linalg.norm(x - self.centroids, axis=1)
            labels.append(np.argmin(distances))
        labels = np.array(labels)
        return labels

import numpy as np
import matplotlib.pyplot as plt

class Neural_Network(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        """Initialization of the neural network parameters"""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))                    
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))                    
      
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_prime(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))   #Numerical stability trick
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        """Forward pass"""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.output = self.softmax(self.z2)
        return self.output
    
    def compute_loss(self, y_true, y_pred):
        """Cross-entropy loss computation"""
        correct_probs = np.sum(y_true*y_pred, axis = 1)  #Probability of the correct class
        loss = -np.mean(np.log(correct_probs + 1e-15))
        return loss
    
    def backward(self, X, y_true, y_pred):
        """Backward pass"""
        N = X.shape[0]
        delta2 = (y_pred - y_true)/N
        
        self.dw2 = np.dot(self.a1.T, delta2)
        self.db2 = np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = np.dot(delta2, self.w2.T)*self.relu_prime(self.z1)
        
        self.dw1 = np.dot(X.T, delta1)
        self.db1 = np.sum(delta1, axis=0, keepdims=True)
        
    def update_parameters(self):
        """Weights and bias update"""
        self.w1 -= self.learning_rate*self.dw1
        self.b1 -= self.learning_rate*self.db1
        self.w2 -= self.learning_rate*self.dw2
        self.b2 -= self.learning_rate*self.db2
        
    def train(self, X_train, y_train):
        """Training loop"""
        loss_tab = []
        for epoch in range(self.epochs):
            y_pred = self.forward(X_train)
            loss = self.compute_loss(y_train, y_pred)
            self.backward(X_train, y_train, y_pred)
            self.update_parameters()
            loss_tab.append(loss)
        return loss_tab
        
    def predict(self, X):
        """Prediction function"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
   
    def display(self, ax=None, left=.1, right=.9, bottom=.1, top=.9, neuron_colors=None, layer_names=None):
        '''
        Affiche l'architecture du réseau de neurones avec matplotlib.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.axis('off')

        layer_sizes = [self.input_size, self.hidden_size, self.output_size]
        n_layers = len(layer_sizes)

        # Valeurs par défaut
        if neuron_colors is None:
            neuron_colors = ['lightblue', 'lightgreen', 'orange']
        if layer_names is None:
            layer_names = ["Input", "Hidden", "Output"]

        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / float(n_layers - 1)

        # Tracé des neurones
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size):
                circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), 
                                    v_spacing / 4., color=neuron_colors[n], ec='k', zorder=4)
                ax.add_artist(circle)
            ax.text(n * h_spacing + left, top + v_spacing / 2, layer_names[n], 
                    fontsize=12, ha='center', va='top', fontweight='bold')

        # Tracé des arêtes (poids, connexions)
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                      [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], 
                                      c='k', alpha=0.5, zorder=1)
                    ax.add_artist(line)

        plt.show()

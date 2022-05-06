import numpy as np

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def accuracy(Y, Y_pred):
    """
    Y: vector of true value
    Y_pred: vector of predicted value
    """
    def _to_binary(x):
        return 1 if x > .5 else 0

    assert Y.shape[0] == 1
    assert Y.shape == Y_pred.shape
    Y_pred = np.vectorize(_to_binary)(Y_pred)
    acc = float(np.dot(Y, Y_pred.T) + np.dot(1 - Y, 1 - Y_pred.T))/Y.size
    return acc
class ShallowNN:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.params = {}
        self.cache = {}
        self.grads = {}
        
    def compute_loss(self, Y, Y_hat):
        """
        Y: vector of true value
        Y_hat: vector of predicted value
        """
        assert Y.shape[0] == 1
        assert Y.shape == Y_hat.shape
        m = Y.shape[1]
        s = Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)
        loss = -np.sum(s) / m
        return loss
    
    
    def init_weights(self):
        self.params['W1'] = np.random.randn(self.n_hidden, self.n_input) * 0.01
        self.params['b1'] = np.zeros((self.n_hidden, 1))
        self.params['W2'] = np.random.randn(self.n_output, self.n_hidden) * 0.01
        self.params['b2'] = np.zeros((self.n_output, 1))
    
    
    def forward(self, X):
        """
        X: need to have shape (n_features x m_samples)
        """
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        A0 = X

        Z1 = np.dot(W1, A0) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        self.cache['Z2'] = Z2
        self.cache['A2'] = A2
     
    
    def backward(self, X, Y):
        """
        [From coursera deep-learning course]
        params: we initiate above with W1, b1, W2, b2
        cache: the intermediate calculation we saved with Z1, A1, Z2, A2
        X: shape of (n_x, m)
        Y: shape (n_y, m)
        """

        m = X.shape[1]

        W1 = self.params['W1']
        W2 = self.params['W2']
        A1 = self.cache['A1']
        A2 = self.cache['A2']

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"dW1": dW1,
                      "db1": db1,
                      "dW2": dW2,
                      "db2": db2}

        
    def get_batch_indices(self, X_train, batch_size):
        n = X_train.shape[0]
        indices = [range(i, i+batch_size) for i in range(0, n, batch_size)]
        return indices
    
    
    def update_weights(self, lr):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        dW1, db1, dW2, db2 = self.grads['dW1'], self.grads['db1'], self.grads['dW2'], self.grads['db2']
        self.params['W1'] -= dW1
        self.params['W2'] -= dW2
        self.params['b1'] -= db1
        self.params['b2'] -= db2
    
    
    def fit(self, X_train, y_train, batch_size=32, n_iterations=100, lr=0.01):
        self.init_weights()
        
        indices = self.get_batch_indices(X_train, batch_size)
        for i in range(n_iterations):
            for ind in indices:
                X = X_train[ind, :].T
                Y = y_train[ind].reshape(1, batch_size)
                
                self.forward(X)
                self.backward(X, Y)
                self.update_weights(lr)
            
            if i % 10 == 0:
                Y_hat = self.cache['A2']
                loss = self.compute_loss(Y, Y_hat)
                print(f'iteration {i}: loss {loss}')
            
            
    def predict(self, X):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        A0 = X

        Z1 = np.dot(W1, A0) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        return A2

    
from sklearn import datasets

X, y = datasets.make_classification(n_samples=5000, random_state=123)

X_train, X_test = X[:4000], X[4000:]
y_train, y_test = y[:4000], y[4000:]

print('train shape', X_train.shape)
print('test shape', X_test.shape)

model = ShallowNN(20, 10, 1)

model.fit(X_train, y_train, batch_size=100, n_iterations=300, lr=0.01)

y_preds = model.predict(X_test.T)

acc = accuracy(y_test.reshape(1, -1), y_preds)
print(f'accuracy: {acc*100}%')
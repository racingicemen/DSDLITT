import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import get_normalized_data

class HiddenLayer(object):
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2

        W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN(object):
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep

    def fit(self, X, Y, lr=1e-6, mu=0.99, decay=0.999, epochs=300, batch_sz=100, show_fig=False):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid = X[-1000:] # last 1000
        Yvalid = Y[-1000:] # last 1000
        X = X[:-1000] # all but the last 1000
        Y = Y[:-1000] # all but the last 1000

        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2)
            self.hidden_layers.append(h)
            M1 = M2
        W = np.random.randn(M1, K) / np.sqrt(M1 + K)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        logits = self.forward(inputs)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)
        prediction = self.predict(inputs)

        n_batches = int(N / batch_sz)
        costs = []
        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

                    session.run(train_op, feed_dict={inputs: Xbatch, labels: Ybatch})

                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={inputs: Xvalid, labels: Yvalid})
                        p = session.run(prediction, feed_dict={inputs: Xvalid})
                        costs.append(c)
                        e = error_rate(Yvalid, p)
                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        Z = tf.nn.dropout(Z, self.dropout_rates[0])
        for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
            Z = h.forward(Z)
            Z = tf.nn.dropout(Z, p)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)

def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a > 0)

def main():
    X, Y = get_normalized_data()

    ann = ANN([500, 300], [0.8, 0.5, 0.5])
    ann.fit(X, Y, epochs=50, show_fig=True)

if __name__ == '__main__':
    main()

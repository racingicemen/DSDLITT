import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.utils import shuffle

from util import get_normalized_data

class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2

        W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)

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

        self.rng = RandomStreams()

        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W = np.random.randn(M1, K) / np.sqrt(M1 + K)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        cache = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY_train = self.forward_train(thX)

        # this cost is for training
        cost = -T.mean(T.log(pY_train[T.arange(thY.shape[0]), thY]))

        updates = [
            (c, decay*c + (1-decay)*T.grad(cost, p)*T.grad(cost, p)) for p, c in zip(self.params, cache)
        ] + [
            (p, p + mu*dp - lr*T.grad(cost, p)/T.sqrt(c+1e-9)) for p, c, dp in zip(self.params, cache, dparams)
        ] + [
            (dp, mu*dp - lr*T.grad(cost, p)/T.sqrt(c+1e-9)) for p, c, dp in zip(self.params, cache, dparams)
        ]

        train_op = theano.function(
            inputs = [thX, thY],
            updates=updates
        )

        pY_predict = self.forward_predict(thX)
        cost_predict = -T.mean(T.log(pY_predict[T.arange(thY.shape[0]), thY]))
        prediction = self.predict(thX)
        cost_predict_op = theano.function(
            inputs = [thX, thY],
            outputs = [cost_predict, prediction]
        )

        n_batches = int(N / batch_sz)
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

                train_op(Xbatch, Ybatch)

                if j % 20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward_train(self, X):
        Z = X
        for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
            mask = self.rng.binomial(n=1, p=p, size=Z.shape)
            Z = mask*Z
            Z = h.forward(Z)

        mask = self.rng.binomial(n=1, p=self.dropout_rates[-1], size=Z.shape)
        Z = mask * Z
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def forward_predict(self, X):
        Z = X
        for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
            Z = h.forward(p * Z)
        return T.nnet.softmax((self.dropout_rates[-1] * Z).dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward_predict(X)
        return T.argmax(pY, axis=1)

def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a > 0)

def main():
    X, Y = get_normalized_data()

    ann = ANN([500, 300], [0.8, 0.5, 0.5])
    ann.fit(X, Y, epochs=2, show_fig=True)

if __name__ == '__main__':
    main()

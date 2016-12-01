import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


from sklearn.utils import shuffle
from util import getData, getBinaryData, error_rate, relu, init_weight_and_bias

class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2

        W, b = init_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)

class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, lr=1e-6, mu=0.99, decay=0.999, reg=1e-11, eps=1e-9, epochs=300, batch_sz=100, show_fig=False):
        lr = np.float32(lr)
        mu = np.float32(mu)
        decay = np.float32(decay)
        reg = np.float32(reg)
        eps = np.float32(eps)

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
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = init_weight_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
        cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)
        cost_predict_op = theano.function(inputs = [thX, thY], outputs = [cost, prediction])

        updates = [
            (c, decay*c + (np.float32(1)-decay)*T.grad(cost, p)*T.grad(cost, p)) for p, c in zip(self.params, cache)
        ] + [
            (p, p + mu*dp - lr*T.grad(cost, p)/T.sqrt(c+eps)) for p, c, dp in zip(self.params, cache, dparams)
        ] + [
            (dp, mu*dp - lr*T.grad(cost, p)/T.sqrt(c+eps)) for p, c, dp in zip(self.params, cache, dparams)
        ]

        train_op = theano.function(
            inputs = [thX, thY],
            updates=updates
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

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return T.argmax(pY, axis=1)

def main():
    X, Y = getData()

    model = ANN([2000, 1000])
    model.fit(X, Y, epochs=2, show_fig=True)

if __name__ == '__main__':
    main()

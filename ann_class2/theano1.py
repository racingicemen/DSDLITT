import theano.tensor as T
import theano
import numpy as np

c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')

# define matrix multiplication
w = A.dot(v)

matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

A_val = np.array([[1,2],[3,4]])
v_val = np.array([5,6])

w_val = matrix_times_vector(A_val, v_val)
print(w_val)

x = theano.shared(20.0, 'x') # initial value, name

cost = x*x + 2*x + 1 # minimum cost 0, at x = -1

# rule for updating x
x_update = x - 0.3*T.grad(cost, x) # derivative of cost w.r.t. x

train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])

for i in range(25):
    cost_val = train()
    print(i, cost_val, x.get_value())

print(x.get_value())

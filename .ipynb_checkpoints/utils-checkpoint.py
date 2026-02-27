import copy
import numpy as np

def cost_function(x,y,w,b,m):
    sqaured_error = 0
    for i in range(m):
        y_pred = np.dot(w,x[i]) + b
        error = y_pred - y[i]
        sqaured_error+=np.square(error)
    return  sqaured_error/(2*m)




def compute_derivative(x, y, w, b, m, n):
    dj_db = 0
    dj_dw = np.zeros(
        n)  # the number of the partial derivatives must be equal to the number of weights as each weight must have their own partial derivative
    for i in range(m):
        y_pred = np.dot(w, x[i]) + b
        error = y_pred - y[i]
        for j in range(n):  # as there are n number of features so the partial derivative depend upon these n features
            dj_dw[j] += error * x[i, j]
        dj_db += error
    return dj_dw / m, dj_db / m



def gradient_descent(x,y,w_init,b_init,gradient,cost_function,alpha,iterations):
    w= copy.deepcopy(w_init)
    b=b_init
    m,n=x.shape
    loss_error = []
    hist_iterations=[]
    for i in range(iterations):
        dj_dw,dj_db = gradient(x,y,w,b,m,n)
        cost = cost_function(x,y,w,b,m)
        loss_error.append(cost)
        hist_iterations.append(i)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
    return w,b,loss_error,hist_iterations
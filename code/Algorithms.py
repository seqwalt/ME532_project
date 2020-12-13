import numpy as np
from sklearn.svm import LinearSVC
# ---------------------------- #
# --- Algorithms Functions --- #
# ---------------------------- #

# --- Ridge Regression Function --- #
def Ridge( A, d, la_array ):
    cols = A.shape[1]
    I = np.eye(cols)
    W = np.empty((cols,len(la_array)))
    for i, λ in enumerate(la_array):
        w = np.linalg.inv(A.T@A + λ*I)@A.T@d
        w = w[:,0]
        W[:,i] = w # each column is a different w
    return W

# --- LASSO Function --- #
def LASSO( A, d, la_array ):
    # Minimize |Ax-d|_2^2 + lambda*|x|_1 (Lasso regression)
    # using iterative soft-thresholding with hot-start.
    max_iter = 10**4
    tol = 10**(-3)
    tau = 1/np.linalg.norm(A,2)**2
    n = A.shape[1]
    num_lam = len(la_array)
    W = np.zeros((n, num_lam))
    w = np.zeros((n,1))
    for i, λ in enumerate(la_array):
        for j in range(max_iter):
            z = w - tau*(A.T@(A@w-d))
            w_old = w
            w = np.sign(z) * np.clip(np.abs(z)-tau*λ/2, 0, np.inf)
            W[:, i:i+1] = w
            if np.linalg.norm(w - w_old) < tol:
                break
    return W

# --- SVM Function: Not used --- #
def my_SVM( A, d, la_array ):
    rows, cols = A.shape
    max_iter = 3*10**5
    #tol = 0.001
    tau = 1/np.linalg.norm(A,2)**2
    #tau = 1e-7
    tol = 5e-5
    num_rand = 1 # number of random points per iteration
    W = np.zeros((cols, len(la_array)))
    w = np.zeros((cols,1))
    i_k = np.arange(rows).reshape(1,-1) # Full GD
    for k, λ in enumerate(la_array):
        #w = np.zeros((cols,1))
        for j in range(max_iter):
            #i_k = np.random.randint(0,rows,size=(1,num_rand)).reshape(1,-1) # SGD
            dAw = (d[i_k,:]*A[i_k,:]@w).reshape(1,-1)
            i_loss = i_k[dAw < 1]
            sum_subgrad = np.sum((-d[i_loss]*A[i_loss,:]).T,axis=1).reshape(-1,1)

            grad = sum_subgrad + 2*λ*w
            w_old = w
            #tau = 1e-4/(j+1)
            w = w - tau*grad
            w = w[:,0]
            if np.linalg.norm(w - w_old) < tol:
                break
        W[:, k] = w
    return W

# --- SVM Function: Used --- #
# Train classifier using linear SVM from SK Learn library
def SVM( A, d, la_array ):
    cols = A.shape[1]
    W = np.zeros((cols, len(la_array)))
    for k, λ in enumerate(la_array):
        clf = LinearSVC( loss='hinge',random_state=0, fit_intercept=True, tol=1e-2, C=1/λ , max_iter=1e5)
        clf.fit(A, np.squeeze(d))
        w_SVM = clf.coef_.T
        W[:,k] = w_SVM[:,0]
    return W

# --- Neural Network --- #
def NN( A_v, A_t, d_t ):
    rows, cols = A_t.shape
    q = 1 #number of classification problems
    M = 400 #number of hidden nodes
    d_T = np.zeros(( len(d_t), 1 ))
    d_T[ d_t == 1 ] = 1

    ## initial weights
    V = np.random.randn(M+1, q);
    W = np.random.randn(cols, M);

    alpha = 0.1 #step size
    L = 20 #number of epochs

    def logsig(_x):
        return 1/(1+np.exp(-_x))

    for epoch in range(L):
        ind = np.random.permutation(rows)
        for i in ind:
            # Forward-propagate
            H = logsig(np.hstack((np.ones((1,1)), A_t[[i],:]@W)))
            Yhat = logsig(H@V)
             # Backpropagate
            delta = (Yhat-d_T[[i],:])*Yhat*(1-Yhat)
            Vnew = V-alpha*H.T@delta
            gamma = delta@V[1:,:].T*H[:,1:]*(1-H[:,1:])
            Wnew = W - alpha*A_t[[i],:].T@gamma
            V = Vnew
            W = Wnew
        # print(epoch)
    # Predicted labels on validation data
    H = logsig(np.hstack((np.ones((A_v.shape[0],1)), A_v@W)))
    dhat = 2*logsig(H@V)-1 # has +1/-1 range
    return dhat

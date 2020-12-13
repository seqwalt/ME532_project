import numpy as np
# ------------------------------------ #
# --- Data Preprocessing Functions --- #
# ------------------------------------ #

# --- One-hot encoding --- #
# This function turns categorical features of n categories, into n features,
# where each new feature is a '1' if the data point is in that category, or
# a '0' if its not in that category. It is called one-hot encoding, because
# there is one '1', and the rest are '0's.
# Note: col_ind -> list of column values (locations of categorical features)
def one_hot( A, col_indices ):
    notOneHot = np.delete(A, col_indices, axis=1)
    for i, cat_col in enumerate(col_indices):
        for j, uniq in enumerate(np.unique(A[:,cat_col])):
            hot_col = -1*np.ones((A.shape[0],1))
            hot_col[A[:,cat_col] == uniq] = 1
            if j == 0:
                hot_cols = hot_col
            else:
                hot_cols = np.hstack((hot_cols,hot_col))
        if i == 0:
            OneHot = hot_cols
        else:
            OneHot = np.hstack((OneHot,hot_cols))
    A_new = np.hstack((notOneHot,OneHot))
    return A_new

# --- Feature Scaling --- #
# Normalize the range of the features by subtracting the mean, then dividing
# by the standard deviation. This is done for each feature.
def normalize( A ):
    ones_col = np.ones((A.shape[0],1))
    u = np.mean(A,axis=0).reshape(1,-1)
    u_mat = ones_col@u
    Au = A - u_mat
    std = np.std(A,axis=0).reshape(1,-1)
    std_mat = ones_col@std
    Asc = Au/std_mat
    #ran = (np.max(A,axis=0) - np.min(A,axis=0)).reshape(1,-1)
    #ran_mat = ones_col@ran
    #Asc = Au/ran_mat
    return Asc

# --- Holdout Indices --- #
# Split the data into 'num_sets' approximately equal sized sets.
def holdout_indices( A, num_sets ):
    rows = A.shape[0]
    remainder = rows%num_sets
    set_size = (rows-remainder)/num_sets
    ind = []
    for i in range(num_sets+1):
        if i <= num_sets - remainder:
            ind = np.append(ind,set_size*i)
        else:
            ind = np.append(ind,set_size + 1 + ind[-1])
    return ind

# --- Training and Validation Sets --- #
# Create the ith set of training and validation data, which is taken from
# the full data using indices 'ind'
def training_validation_sets( A, d, ind, i ):
    A_t = np.vstack(( A[int(ind[0]):int(ind[i]),:] , A[int(ind[i+1]):int(ind[-1]),:] )) # training data
    A_v = A[int(ind[i]):int(ind[i+1]),:] # validation data
    d_t = np.vstack(( d[int(ind[0]):int(ind[i]),:] , d[int(ind[i+1]):int(ind[-1]),:] )) # training labels
    d_v = d[int(ind[i]):int(ind[i+1]),:] # validation labels
    return [A_t, A_v, d_t, d_v]

# --- Low rank approximation --- #
def low_rank( A, rank ):
    Ar = np.zeros((A.shape))
    U,s,VT = np.linalg.svd(A,full_matrices=False)
    for i in range(rank):
        Ui = U[:,i].reshape(-1,1)
        VTi = VT[i,:].reshape(1,-1)
        Ar = Ar + s[i]*Ui@VTi
    U,s,VT = np.linalg.svd(Ar,full_matrices=False)
    s = s[np.log10(s) > -1]
    return [Ar, s]

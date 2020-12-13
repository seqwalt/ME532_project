import numpy as np
from numpy import linalg as LA
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from DataPreprocess import one_hot, normalize, holdout_indices, training_validation_sets, low_rank
from ErrorAnalysis import error_metrics
from BestWeights import best_weights
from Algorithms import Ridge, LASSO, SVM, NN

# --- Load data --- #
A_d = np.genfromtxt('../data/cardio_disease.csv',delimiter=';',skip_header=1)
A = A_d[:,0:11]
d = A_d[:,11].reshape(-1,1)
d[d == 0] = -1

# --- Data Preprocessing --- #
#Feature scaling
A = normalize(A)
# one-hot encoding for gender, cholesterol, glucose, smoke, alcohol and
# physical activity
A = one_hot(A,[1,6,7,8,9,10])

# Check Singular Values
U,s,VT = np.linalg.svd(A,full_matrices=False)

# Low-rank approx of A
rank = 14
Ar, s = low_rank(A, rank)

# --- Cross Validation with all Algorithms --- #
num_class = 4 # number of classifiers
num_sets = 10 # 10-fold cross validation
ind = holdout_indices(Ar,num_sets) # holdout indices
ERS = np.empty((num_sets,num_class))
ERR = np.empty((num_sets,num_class))
SQ_ER = np.empty((num_sets,num_class))
LAMBDA = np.empty((num_sets,3))

#lam_valsRID = np.geomspace(1e-3,1e6,num=100) # logarithmically spaced values
#lam_valsLAS = np.geomspace(1e-3,1e6,num=100)
#lam_valsSVM = np.geomspace(1e-3,1e3,num=100)
lam_valsRID = [8470.50]
lam_valsLAS = [419.20]
lam_valsSVM = [1.060]

# Loop through the validation sets
for i in range(num_sets):
    print('holdout set: ',i+1)
    A_T, A_V, d_t, d_v = training_validation_sets(Ar,d,ind,i)
    A_t = np.hstack(( np.ones((A_T.shape[0],1)), A_T ))
    A_v = np.hstack(( np.ones((A_V.shape[0],1)), A_V ))
    # Split validation set into two roughly equal sizes
    half = int(np.round(A_v.shape[0]/2))
    A_v1 = A_v[0:half,:]; d_v1 = d_v[0:half,:]
    A_v2 = A_v[half::,:]; d_v2 = d_v[half::,:];

    # Best weights, and corresponding indices
    wR, indR = best_weights( Ridge(A_t,d_t,lam_valsRID),A_v1,d_v1 ) # best Ridge regression weights among lam_vals
    wL, indL = best_weights( LASSO(A_t,d_t,lam_valsLAS),A_v1,d_v1 ) # best LASSO weights among lam_vals
    wS, indS = best_weights( SVM(A_t,d_t,lam_valsSVM),A_v1,d_v1 ) # best SVM weights among lam_valsSVM
    d_predNN = NN(A_v2,A_t,d_t) # Neural network predictions
    D_pred = np.hstack(( A_v2@wR, A_v2@wL, A_v2@wS, d_predNN ))

    # Error Metric
    ers, err, sq_er = error_metrics(D_pred,d_v2)
    ERS[i,:] = ers
    ERR[i,:] = err
    SQ_ER[i,:] = sq_er

    # Best Lambda Values
    LAMBDA[i,:] = [lam_valsRID[int(indR[0])], lam_valsLAS[int(indL[0])], lam_valsSVM[int(indS[0])]]
print('Done')

# --- View Errors --- #
avg_ers = np.sum(ERS,axis=0)/num_sets
avg_err = np.sum(ERR,axis=0)/num_sets
avg_sqr = np.sum(SQ_ER,axis=0)/num_sets
lambdas  = np.sum(LAMBDA,axis=0)/num_sets

print('Average squared error:')
print('  Ridge: ',avg_sqr[0])
print('  LASSO: ',avg_sqr[1])
print('    SVM: ',avg_sqr[2])
print('     NN: ',avg_sqr[3])
print()

print('Average number of errors:')
print('  Ridge: ',avg_ers[0])
print('  LASSO: ',avg_ers[1])
print('    SVM: ',avg_ers[2])
print('     NN: ',avg_ers[3])
print()

print('Average error rate:')
print('  Ridge: ',avg_err[0])
print('  LASSO: ',avg_err[1])
print('    SVM: ',avg_err[2])
print('     NN: ',avg_err[3])
print()

print('Best Average λ:')
print('  Ridge: ',lambdas[0])
print('  LASSO: ',lambdas[1])
print('    SVM: ',lambdas[2])

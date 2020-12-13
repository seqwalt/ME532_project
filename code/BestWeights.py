import numpy as np
from numpy import linalg as LA
# --- Best Weights Function --- #
# Returns the best weight vector. The best weight vector has the least number
# of errors. If multiple vectors are tied for best, the vector with the
# smallest L2 norm is chosen. This function as returns the index at which the
# best weight was found, so the corresponding lambda value can be found.
def best_weights( W, A_v, d_v ):
    # min weight of weights with most correct classifications
    errors = np.count_nonzero(np.sign(A_v@W) - d_v,axis=0)
    w_best = W[:,errors == np.min(errors)] # pick column with least num of errors

    w_norms = LA.norm(w_best,axis=0)
    w_Best = w_best[:,w_norms == np.min(w_norms)]
    w_BEST = w_Best[:,0].reshape(-1,1)

    INDEX = np.argwhere((w_BEST == W))
    index = INDEX[0,1]*np.ones((len(w_BEST),1))

    return [w_BEST, index]

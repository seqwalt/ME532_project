import numpy as np
# -------------------------------- #
# --- Error Analysis Functions --- #
# -------------------------------- #

# --- Number of Errors --- #
def errors( d_pred, d_test ):
    ers = np.count_nonzero(np.sign(d_pred) - d_test,axis=0)
    return ers

# --- Error Rate --- #
def error_rate( d_pred, d_test ):
    ers = errors(d_pred,d_test)
    err = (1/len(d_test))*ers
    return err

# --- Squared Error --- #
def squared_error( d_pred, d_test ):
    sq_er = np.sum((d_pred - d_test)**2,axis=0)
    return sq_er

# --- Error metrics --- #
def error_metrics( d_pred, d_test ):
    ers = errors( d_pred, d_test )
    err = error_rate( d_pred, d_test )
    sq_er = squared_error( d_pred, d_test )
    return [ers, err, sq_er]

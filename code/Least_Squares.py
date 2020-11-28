import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as img
from DataPrepFuncs import rgb2gray, rock_labels, downsample

first_pic = 1
last_pic = 101

for num_pic in range(first_pic,last_pic+1):
    str_pic = str(num_pic)
    for i in range(4-len(str_pic)):
        str_pic = '0'+str_pic

    os.chdir('../dataset/images')
    render = img.imread('render/render'+str_pic+'.png')
    ground = img.imread('ground/ground'+str_pic+'.png')
    os.chdir('../../Github')

    # Raw Data to grayscale
    raw = rgb2gray(render)
    # Labelled Data
    gro = np.round(ground)
    lab = rock_labels(gro)

    # Reduce dimension
    f = 4 # originally 4
    RAW = downsample(raw,f)
    LAB = downsample(lab,f)
    LAB[LAB < 0] = -1
    LAB[LAB >= 0] = 1

    # Remove n outer layers for feature extraction
    ROW, COL = RAW.shape
    raw_ = RAW
    lab_ = LAB
    n = 4 # how many layers to take off -- originally 3
    for x in range(n):
        row,col = raw_.shape
        rows_off_r = np.delete(raw_, [0,row-1],axis=0)
        raw_  = np.delete(rows_off_r,[0,col-1],axis=1)
        rows_off_l = np.delete(lab_, [0,row-1],axis=0)
        lab_  = np.delete(rows_off_l,[0,col-1],axis=1)
    row, col = raw_.shape

    # Features
    fe1 = raw_.reshape(-1,1) # Value of raw pixel
    fe2 = np.sqrt(fe1)
    fe3 = fe1**2
    fe4 = fe1**3
    fe5 = fe1**4
    A_curr = np.hstack((fe1,fe2,fe3,fe4,fe5))

    for i in range(row):
        for j in range(col):
            r = n + i
            c = n + j
            k = n
            #sur_fea = np.hstack((RAW[r+k,c],RAW[r-k,c],RAW[r,c+k],RAW[r,c-k],RAW[r+k,c+k],RAW[r+k,c-k],RAW[r-k,c+k],RAW[r-k,c-k]))
            sur_fea = np.hstack((RAW[r+1,c],RAW[r-1,c],RAW[r,c+1],RAW[r,c-1],RAW[r+1,c+1],RAW[r+1,c-1],RAW[r-1,c+1],RAW[r-1,c-1]))
            for k in range(2,n+1):
                sur_fea = np.hstack((sur_fea,RAW[r+k,c],RAW[r-k,c],RAW[r,c+k],RAW[r,c-k],RAW[r+k,c+k],RAW[r+k,c-k],RAW[r-k,c+k],RAW[r-k,c-k]))
                if i+j == 0:
                    sur_data = sur_fea
            '''if i+j == 0:
                sur_data = sur_fea'''
            if i+j != 0:
                sur_data = np.vstack((sur_data,sur_fea))

    fe6 = np.ones((row*col,1)) # Column of ones
    A_curr = np.hstack((A_curr,sur_data,fe6))
    #A_curr = np.hstack((A_curr,fe6))
    y_curr = lab_.reshape(-1,1)

    if num_pic == first_pic:
        A = A_curr
        y = y_curr
    elif num_pic != last_pic:
        A = np.vstack((A,A_curr))
        y = np.vstack((y,y_curr))

    if num_pic == last_pic:
        A_test = A_curr
        lab_test = lab_

        fig = plt.figure(figsize=(12,12), dpi= 80)
        ax1 = fig.add_subplot(221);
        ax1.set_title('Raw')
        plt.imshow(raw, cmap = 'gray');

        ax2 = fig.add_subplot(222);
        ax2.set_title('Labelled')
        plt.imshow(lab_, cmap = 'gray');
        fig.add_subplot(223);
        plt.imshow(raw_, cmap = 'gray');

# Naive Least Squares Prediction
#U,s,VT = np.linalg.svd(A,full_matrices=False)
w = np.linalg.inv((A.T@A))@A.T@y
y_pred = np.sign(A_curr@w).reshape(row,col)
#y_pred = (A@w).reshape(row,col)

# Errors
true = lab_test.reshape(-1,1)
per_err = np.count_nonzero(np.sign(A_test@w) + true == 0)/len(true)
print(per_err)
print(w)

# Plot
pred = fig.add_subplot(224);
pred.set_title('Prediction')
plt.imshow(y_pred,cmap='gray');
plt.show()

import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.image as img
from DataPrepFuncs import rgb2gray, rock_labels, downsample

os.chdir('../dataset/images')
render = img.imread('render/render3400.png')
ground = img.imread('ground/ground3400.png')
os.chdir('../../Github')

# Raw Data to grayscale
raw = rgb2gray(render)
# Labelled Data
gro = np.round(ground)
lab = rock_labels(gro)

# View raw and labelled data
fig = plt.figure(figsize=(12,12), dpi= 100)
ax1 = fig.add_subplot(321);
ax1.set_title('Raw')
plt.imshow(raw, cmap = 'gray');

ax2 = fig.add_subplot(322);
ax2.set_title('Labelled')
plt.imshow(lab, cmap = 'gray');

# Reduce dimension
f = 8
raw_ = downsample(raw,f)
lab_ = downsample(lab,f)
lab_[lab_ < 0] = -1
lab_[lab_ >= 0] = 1

fig.add_subplot(323);
plt.imshow(raw_, cmap = 'gray');
fig.add_subplot(324);
plt.imshow(lab_, cmap = 'gray');

# Naive Least Squares Prediction
row, col = raw_.shape
A = raw_.reshape(-1,1)  # Features --> naive feature is value of pixel
y = lab_.reshape(-1,1)
w = np.linalg.inv((A.T@A))@A.T@y
y_pred = np.sign(A@w).reshape(row,col)

pred = fig.add_subplot(326);
pred.set_title('Naive Prediction')
plt.imshow(y_pred,cmap='gray');

plt.show()

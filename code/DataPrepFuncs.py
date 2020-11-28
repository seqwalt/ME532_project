import numpy as np

# Data-Prep Functions

# --- RGB to Grayscale Function --- #
# Gray_scale = R * 299/1000 + G * 587/1000 + B * 114/1000
# https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
def rgb2gray(RGB_img):
    grayscale = np.array([[.0299],[.587],[.114]])
    rows, cols, rgb = np.array(RGB_img.shape)
    gray_img = (RGB_img@grayscale).reshape(rows,cols)
    return gray_img

# --- Rock-Labels-Only Function --- #
# Alters the ground/clean truth images to set all rocks to +1, and everything else to -1
# Assume current RGB values are
# [1,0,0](sky), [0,1,0](small rock), [0,0,1](big rock), [0,0,0](ground)
def rock_labels(truth_img):
    rows, cols, rgb = np.array(truth_img.shape)
    rock_img = (truth_img@np.array([[0],[1],[1]])).reshape(rows,cols)
    rock_img[rock_img == 0] = -1
    rock_img[rock_img != -1] = 1 #necessary for when over-lapping rocks create turquoise
    return rock_img

# --- Downsampling function --- #
# img_in = original grayscale image
# f = downsampling factor (f = 2 --> half as many rows, half as many columns)
# f is an integer, to make downsampling simpler
def downsample(img_in,f):
    if f == 1:
        return img_in
    rows, cols = img_in.shape
    nd = rows/f # choose f s.t. nd and md are integers
    md = cols/f
    n = int(nd)
    m = int(md)
    if n == nd and m == md: # Make sure nd and md have integer values
        down = np.empty((n,m))
        for i in range(n):
            for j in range(m):
                down[i,j] = np.mean(img_in[f*i : f*i+f-1, f*j : f*j+f-1])
        return down
    else:
        print('n and m must be integers')
        return np.zeros((n,m))

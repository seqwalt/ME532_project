# Cardiovascular Disease Classification

This is a project for ECE 532: Matrix Methods in Machine Learning at the University of Wisconsin, Madison. The purpose is to perform a classification task on a public dataset.

## Dataset: [Artificial Lunar Landscape Dataset](https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset)

This dataset provides semantic segmentation of photo-realistic lunar scenes. The scenes are segmented into three classes of small rocks, large rocks, and sky. This dataset contains 9,766 rendered and labelled scenes, and images of the actual lunar surface for testing. There is a ground truth and a "cleaned" ground truth (rocks with most pixels) for each render. The pixel dimension of each image is 480 by 720.

![data_sample](https://github.com/seqwalt/Moon-Rock-Detection/blob/master/media/data_sample.png)

## Algorithms: Least Squares (LS), Support Vector Machine (SVM), Neural Network (NN)

*Data Preprocessing*: This project is a classification task, in the sense that each pixel of the image will be classified as sky, large rock, or small rock. I will create features to try to provide enough information to classify each pixel. One example of a feature of a pixel could be the average value of the immediate neighboring pixels. PCA will then be used to determine the most important features, and will highlight features that can be removed. I will iteratively design features and test them with a simple LS classifier until performance has the lowest possible cross validation error. This feature engineering will be a crucial step toward accurate classification.

*Classification*: The feature data will be used in LS and SVM classifiers.  For the LS method,  I may have to consider regularization if there is too much linear dependance within the feature set.  I will compare these results with a neural network approach, where a convolutional neural network (CNN) will be used. CNNs are the standard approach to semantic segmentation, and do not require feature selection. The CNN is expected to have the best result of all the classifiers.  Cross validation will be used to measure success.

## First Update

See `data_prep.ipynb` to view data preprocessing functions. Least squares is implemented in the `Least_Squares.py` file, which currently uses `DataPrepFuncs.py` to import data preprocessing functions.

## Timeline:
`10/22/2020:` **Proposal**  
`11/01/2020:` Be able to manipulate data in python  
`11/10/2020:` Do feature engineering alongside PCA  
`11/14/2020:` Apply LS classification  
`11/17/2020:` **First update**  
`11/25/2020:` SVM classification  
`12/01/2020:` **Second Update**  
`12/05/2020:` Implement CNN  
`12/12/2020:` **Final Project Report**  

# Cardiovascular Disease Classification

This is a project for ECE 532: Matrix Methods in Machine Learning at the University of Wisconsin, Madison. The purpose is to perform a classification task on a public dataset.

## Usage:

To train the binary classifiers at once, run `RunAlgorithms.py` while maintaining the same folder-file structure. Run `CreatePlots.py` to recreate the correlation heatmap and singular value plots. Additionally, the jupyter notebook I used to create and tinker with the algorithms, `sandbox.ipynb`, creates singular value plots, trains the binary classifiers, and is self-contained so it does not have dependencies on other files.

## Dataset: [Cardiovascular diseases dataset (clean)](https://www.kaggle.com/aiaiaidavid/cardio-data-dv13032020)

In addition to including whether a patient has a CVD, this dataset provides 11 features: age, height, weight, gender, systolic blood pressure, diastolic blood pressure, cholesterol, glucose, smoking level, alcohol drinking level and physical activity.  Six of the eleven of these features are categorical including gender, cholesterol, glucose, smoke, alcohol and physical activity. It includes data from 68,783 patients, and the dataset creator has ensured there are no missing or incorrect values. Here is a correlation heatmap of the original features creates with `pandas`:

<img src="https://github.com/seqwalt/ME532_project/blob/master/media/heatmap.png" alt="correlation heatmap" width="900">

## Algorithms: Least Squares (LS), Support Vector Machine (SVM), Neural Network (NN)

The feature data will be used in LS, SVM and NN classifiers. I have used 10-fold cross validation (9 training sets, and 1 test set) to compare the result of Ridge regression, LASSO, SVM and NN. The features matrix used for classification uses one-hot encoding for the categorical features and feature scaling to normailize the range of the features. The current analysis is in the following table:

|           | Avg Sq Error | Avg Num Errors | Avg Error Rate | Best Avg λ |
| --------- | ------------ | -------------- | -------------- | ---------- |
| **Ridge** |       2669.2 |          946.6 |          0.275 |    8470.5  |
| **LASSO** |       2665.1 |          945.8 |          0.275 |     419.2  |
|   **SVM** |       5189.8 |          942.7 |          0.274 |       1.06 |
|    **NN** |       3282.8 |         1042.8 |          0.303 |     ---    |

I addition to using one-hot encoding and feature scaling, a low-rank approximations is used to bring the one-hot encoded matrix from a rank of 19 to that of 14. The log of the singular values were plotted, showing the change of the singular value from before and after normalizing, one-hot encoding and rank-approximating the feature matrix. Notice that the normalization has caused the singular values to even out.

<img src="https://github.com/seqwalt/ME532_project/blob/master/media/singular_vals.png" alt="singular values" width="800">

## Timeline:
`11/24/2020:` Switched projects  
`11/25/2020:` LS and SVM classification  
`12/01/2020:` **Second Update**  
`12/05/2020:` Implement NN  
`12/12/2020:` **Final Project Report**  

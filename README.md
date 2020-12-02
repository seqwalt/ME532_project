# Cardiovascular Disease Classification

This is a project for ECE 532: Matrix Methods in Machine Learning at the University of Wisconsin, Madison. The purpose is to perform a classification task on a public dataset.

## Dataset: [Cardiovascular diseases dataset (clean)](https://www.kaggle.com/aiaiaidavid/cardio-data-dv13032020)

This dataset is used top predict if a patient has cardiovascular disease. I provides 11 features: age, height, weight, gender, systolic blood pressure, diastolic blood pressure, cholesterol, glucose, smoke, alcohol, and physical activity. Six of these features are categorical, which will require the use of one-hot encoding of these features. 

## Algorithms: Least Squares (LS), Support Vector Machine (SVM), Neural Network (NN)

The feature data will be used in LS, SVM and NN classifiers. I have used 10-fold cross validation (9 training sets, and 1 test set) to compare the result of Ridge regression, LASSO and SVM. I plan on using a NN as well, which is expected to have the best result of all the classifiers because it can approximate any classifier boundary. The current preliminary results do not incorporate one-hot encoding, and currently SVM has the worst performance in terms of correct classifications likely due to one-hot not being used. The current analysis is in the following table:

|     | Avg Sq Error | Avg Num Errors | Avg Error Rate | Best Avg λ |
| --- | ------------ | -------------- | -------------- | ---------- |
**Ridge** | 5322.1 | 1886.1 | 0.274 | 22.3
**LASSO** | 6854.8 | 2269.0 | 0.330 | 1873817.4
**SVM**   | 6880.7 | 3404.1 | 0.495 | 0.0

I addition to using one-hot encoding, I plan to test if any low-rank approximations of the data result in better classifiers than the full-rank data. The log of the singular values were plotted, and it looks like a rank-5 matrix would be a good initital test, because the first 5 singular values are clearly seperate from the others. Here is the plot:


## Timeline:
`11/24/2020:` Switched projects  
`11/25/2020:` LS and SVM classification  
`12/01/2020:` **Second Update**  
`12/05/2020:` Implement CNN  
`12/12/2020:` **Final Project Report**  

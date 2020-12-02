# Cardiovascular Disease Classification

This is a project for ECE 532: Matrix Methods in Machine Learning at the University of Wisconsin, Madison. The purpose is to perform a classification task on a public dataset.

## Dataset: [Cardiovascular diseases dataset (clean)](https://www.kaggle.com/aiaiaidavid/cardio-data-dv13032020)

This dataset is used top predict if a patient has cardiovascular disease. I provides 11 features: age, height, weight, gender, systolic blood pressure, diastolic blood pressure, cholesterol, glucose, smoke, alcohol, and physical activity. Six of these features are categorical, which will require the use of one-hot encoding of these features. 

## Algorithms: Least Squares (LS), Support Vector Machine (SVM), Neural Network (NN)

The feature data will be used in LS, SVM and NN classifiers. I have used 10-fold cross validation (9 training sets, and 1 test set) to compare the result of Ridge regression, LASSO and SVM. I plan on using a NN as well, which is expected to have the best result of all the classifiers because it can approximate any classifier boundary. The current preliminary results do not incorporate one-hot encoding, so currently SVM has the worst performance in terms of correct classifications.

## Timeline:
`11/24/2020:` Switched projects  
`11/25/2020:` LS and SVM classification  
`12/01/2020:` **Second Update**  
`12/05/2020:` Implement CNN  
`12/12/2020:` **Final Project Report**  

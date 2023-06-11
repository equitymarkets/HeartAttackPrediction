# health_project_group_1

## we examine heart attack prediction factors including: 


## Data Collecting (ETL)

## PCA
* Since we are looking at many different factors to predict an outcome, it will be helpful to reduce the dimensions to provide simplicity. This will reduce accuracy but will make it easier to see the approximation of every variable.
* We hope to reduce our data to only two dimensions 
* There is a lot of variance within each of our variables, we want to simplify and create bigger, more meaningful features
* Using PCA we find three clusters, which we can then use to categorize individuals based on certain traits
* We find there are three clusters, which here means groups that have certain characteristics
![bokeh_plot](https://github.com/equitymarkets/health_project_group_1/assets/65323795/e3007c9f-98e7-4e27-a5b2-36f22d4a893e)
* Gives an overall more simplistic view of our data

## Logistic Regression


## Random Forests 
We also used a Random Forest Classifier on the binary data, which is 1 if they had the condition before the heart attack and 0 if they did not. We first ran our model on the baseline hyper-parameters to see where the model would be at.  Below, we show the confusion matrix, accuracy score, and classification report. 

While we do have a high accuracy score, you can see from the confusion matrix that this model is really bad at predicting people with heart attacks, with it missing on 70 people. This can be explained by the fact that the data has a big class imbalance, as people with heart attacks only make up about 5% of our data. With this class imbalance, our model was not learning from the features and treating heart attacks as a rare occurrence. To solve this problem, we decided to down sample our non-heart attack class, so that the heart attack people make up more of the data. We also tuned our hyper-parameters with RandomizedSearchCV() and GridSearchCV() to get the best model for the data. 

First, we downsized the data so that the non-heart attack group and heart attack group had a ratio of 2:1. We, then, split the data into training and testing sets and applied RandomizedSearchCV() with a range of different hyper-parameters. After that, we predicted the training set, and its results are shown below. 

With the parameters from the randomized search, we did a GridSearchCV() with values close to the hyper-parameters of the randomized search to see if we can get a better model. The results from that model is shown below. 

We applied the same process to a downsized data set where the ratio was 3:1 non-heart attack to heart attack, but we found that the smaller down sampled model was better at predicting heart attacks and continue with that model. 

We wanted to check if there was any biases from the model, so we decided to use it to predict the whole data set. The goal is to check if we get the same results from the testing, then we can say that there is no biases in the smaller data set that would skew the model into one direction. The results from predicting the whole data set is shown below. 

The total results shown are similar to the testing results, so we can say that there were no biases in our smaller data set. From here, we want to check with conditions are the best predictors of a heart attack, so we graphed the feature importance. 

From the graph, we can see that heart disease, heart failure, stroke, and angina pectoris are the main conditions used to predict heart attacks. For another visualization, we show the decision graph of our model.

## Neural Networks

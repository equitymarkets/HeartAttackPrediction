# health_project_group_1

## we examine heart attack prediction factors including: 


## Data Collecting (ETL)
We examined National Health and Nutrition Examination Survey (NHANES) data from the Centers for Disease Control and Prevention (CDC). The main data set that we used was Pre-Pandemic data from 2017 to March 2020. While there are large number of surveys to use, we focused our attention primarily on two surveys: Medical Conditions and Demographics. A big challenge with this data and our project overall was the data cleaning process. The medical conditions table contained over 60 unique columns. Since there were so many columns, we first decided which columns were of most importance to our question. Once that was determined, we needed to clean all of the column names, since they were initially unique codes that provided no insight as to what the column was. 

After the data table was in a readable format, our next focused turned to cleaning up the data since it had a lot of missingness. Binary columns were changed to 1 (had the diseases) 0 (did not report having the disease). Cells in columns representing the age at which someone had a specified conditions were converted to 0 if they either did not have the disease or if it fell after the age of the heart attack. A years column for each condition was added to indicate the difference between the condition age and the heart attack age. A max age column was also added to indicate the age of the condition that was closest to but still prior to the heart attack. This cleaned table was outputted to a csv file and used for further analysis.

An additional table was created that contained binary columns to indicate whether or not the patient had the condition prior to the heart attack. This table was also used in the further analyses. 


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

![Screen Shot 2023-06-11 at 4 37 24 PM](https://github.com/equitymarkets/health_project_group_1/assets/114087082/c520714a-703c-4576-8104-885ecddf3073)

While we do have a high accuracy score, you can see from the confusion matrix that this model is really bad at predicting people with heart attacks, with it missing on 70 people. This can be explained by the fact that the data has a big class imbalance, as people with heart attacks only make up about 5% of our data. With this class imbalance, our model was not learning from the features and treating heart attacks as a rare occurrence. To solve this problem, we decided to down sample our non-heart attack class, so that the heart attack people make up more of the data. We also tuned our hyper-parameters with RandomizedSearchCV() and GridSearchCV() to get the best model for the data. 

First, we downsized the data so that the non-heart attack group and heart attack group had a ratio of 2:1. We, then, split the data into training and testing sets and applied RandomizedSearchCV() with a range of different hyper-parameters. After that, we predicted the training set, and its results are shown below. 

![Screen Shot 2023-06-11 at 4 56 07 PM](https://github.com/equitymarkets/health_project_group_1/assets/114087082/a879bb36-721e-48f1-b54f-4692984b5f56)

With the parameters from the randomized search, we did a GridSearchCV() with values close to the hyper-parameters of the randomized search to see if we can get a better model. The results from that model is shown below. 

![Screen Shot 2023-06-11 at 5 05 21 PM](https://github.com/equitymarkets/health_project_group_1/assets/114087082/4296f247-c15d-41e1-a61a-346da915fe31)

We applied the same process to a downsized data set where the ratio was 3:1 non-heart attack to heart attack, but we found that the smaller down sampled model was better at predicting heart attacks and continue with that model. 

We wanted to check if there was any biases from the model, so we decided to use it to predict the whole data set. The goal is to check if we get the same results from the testing, then we can say that there is no biases in the smaller data set that would skew the model into one direction. The results from predicting the whole data set is shown below. 

![Screen Shot 2023-06-11 at 5 06 52 PM](https://github.com/equitymarkets/health_project_group_1/assets/114087082/c02cda9e-5a63-4199-ad31-d44fcec70561)

The total results shown are similar to the testing results, so we can say that there were no biases in our smaller data set. From here, we also wanted to test this model with data from a different year, so we applied the model to data from the years 2015-2016. The results from applying that model is shown below.

![Screen Shot 2023-06-11 at 5 07 57 PM](https://github.com/equitymarkets/health_project_group_1/assets/114087082/e71c0d4f-7357-44cb-b0c8-006901ca60fb)

To check for the conditions most used to predict heart attack, we got the feature importances from our model and made a bar graph. From the graph, we can see that heart disease, heart failure, stroke, and angina pectoris are the main conditions used to predict heart attacks.

![Screen Shot 2023-06-11 at 5 05 37 PM](https://github.com/equitymarkets/health_project_group_1/assets/114087082/82e6eccc-9cb4-4786-be3e-553f5eb2a4ed)

## Neural Networks

For the Neural Network we used the Medical Conditions data, a table containing dozens of conditions ranging from heart disease, stroke, and cancer to asthma, arthrithis, and hay fever. This also proved to be a challenge. As mentioned before, the NaN values were changed to 0 for simplicity purposes. Doing this eased the processing of the data.

The data was first further cleaned, with removal columns that have no value in the analysis, such as the ID number. The get_dummies function was used to turn existing non-binary columns into binary. Then the x and y values were chosen, y being the target, in this case Heart Attack, and X being a set of variables to test for in the model, or features, to try to predict whether or not heart attacks could be predicted. 

The data was then scaled.

After scaling we were ready to set up the model. The amount of layers, neurons in each layer, and activation functions were chosen. 

We then compiled the data. 

Finally, the models were run by choosing the amount of epochs and training the model. 

Upon starting the analysis, excessive accuracy was recorded, with values of up to 100% on the second or third epoch. There seemed to be either data leaking or overfitting, or perhaps there was an error in the model. 

We first tried to analyze the actual inputs by creating a csv and checking the data. We cut additional inputs from the features, to ensure that we were not leaking the data. We inserted a function for early stopping. Still, high levels of accuracy remained. 

![loss](https://github.com/equitymarkets/health_project_group_1/assets/49753517/42fc65fc-d61d-40f3-8da2-77d0ac0eaffb)

![accuracy](https://github.com/equitymarkets/health_project_group_1/assets/49753517/0970a77f-f0c0-4997-a676-f4a2f9f11269)

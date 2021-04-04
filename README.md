
# Capstone Project for the Azure ML Engineer Program by Udacity

The goal of this project is to develop two different ML models using the Azure ML platform, and then, deploying the best one as a web service. First, a Scikit-learn model will be developed and tuned using the Python SDK and HyperDrive respectively. Once this model is optimized, its results are compared with a model generated from the Automated ML Azure service, and the one with the best performance is deployed to the web.

## Project Overview

The process flow followed in the project is illustrated below:

<img src=".\images\Project Workflow.jpg">

## Dataset

### Overview

The dataset used to develop both models comes from Kaggle datasets. It deals with heart failures, bringing information on 12 categorical and numerical features which can be used to forecast mortality rates by cardiovascular diseases (CVDs). Additionally, The dataset has a total of 299 observations.

### Task

As it was mentioned previously, this data is going to be used to predict heart failures caused by CVDs, in order to provide an early detection ML model using the 12 clinical variables included in the dataset. A brief summary of these features is the following:

<img src=".\images\Dataset Features.jpg">

### Access
For training the AutoML model, I registered the dataset from local files using the Datasets Hub in the Azure ML Studio.
<br><img src=".\images\Dataset.jpg"><br>
By contrast, I used the following dataset url from github for training the customised model using HyperDrive: https://github.com/htrismicristo/Capstone-Project-Azure-ML-Engineer-Microsoft-Udacity/blob/main/heart_failure_clinical_records_dataset.csv
<br><img src=".\images\Dataset Exploration.jpg"><br>

## Automated ML
In overview, the following were the main settings and parameters used for the AutoML model on the Azure platform:<br>
* n_cross_validations = 5. It sets the number of cross validations to carry out.
* iterations = 30. It specifies how many algorithm and parameter combinations to test during the experiment.
* max_concurrent_iterations = 4. It represents the maximum number of iterations that could be performed in parallel.
* primary_metric = Accuracy. This is the metric which will be optimized for model training and selection.
* target column = DEATH_EVENT. Whether the patient died from a heart failure or not.
* task = classification. Based on the project's goal and the nature of the data, the task is clearly a classification one.
* experiment_timeout_minutes = 30. It determines the maximum amount of time all iterations can take before the experiment terminates. 

### Results
After running the experiment using AutoML, the best model found was the Voting Ensemble one. This algorithm is known for combining multiple models to enhance its performance. This ensemble model makes use of the weighted average of predicted class probabilities to give a final prediction.<br>
<br>Best run id: <br>
Accuracy: <br>
<br>The model parameters include:<br>
<br>
* Random State = None. To validate our results over multiple runs this value should remain steady.
* Reg_alpha = . To moderate the model predictions we can increase this value.
* Reg_lambda = . The L2 regularization term for tackling overfitting.
* Silent = True.
* Verbose = -10.
* Robustscaler = True.
* Quantile Range = [25, 75].
* with_centering = True.
* with_scaling = False.
<br>
The LightGBM algorithm was surpassed by the VotingEnsemble model, despite having an accuracy of 0.8060.


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response


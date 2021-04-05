
# Capstone Project for the Azure ML Engineer Program

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
* Random State = None. To validate our results over multiple runs this value should remain steady.
* Reg_alpha = . To moderate the model predictions we can increase this value.
* Reg_lambda = . The L2 regularization term for tackling overfitting.
* Silent = True.
* Verbose = -10.
* Robustscaler = True.
* Quantile Range = [25, 75].
* with_centering = True.
* with_scaling = False.
The LightGBM algorithm was surpassed by the VotingEnsemble model, despite having an accuracy of 0.8060.


## Hyperparameter Tuning

### Algorithm
Given the kind of Machine learning problem and the binary nature of the variable of interest, a logistic regression model was chosen. It uses a logistic function to model a binary dependent variable (DEATH_EVENT) and make predictions using conditional probability (1 or 0 given X).

### Hyperparameters
The key Hyperparameters used were:
* the 'C' or inverse regularization parameter: ~uniform(0,1).
* max_iter: [0, 150]. It represents the maximum number of iterations allowed. 

### Parameter sampler
The RandomParameterSampler was used since it uses less computational resources to randomly find the best parameters from a large sample space. In other words, it provides better results in a smaller amount of time.

### Termination Policy
A BanditPolicy was used to improve the computational efficiency, terminating early the poorly performing runs. The following were its main parameters:<br>
* The evaluation interval = 2. It determines the frequency in which the policy will be applied.
* The slack factor = 0.1. It specifies the allowed distance from the best performing run. The runs whose best metric is less than (primary metric of best performing run at given interval/(1+slack factor)) will be terminated.


### Results
After running the experiment using hyperdrive and the hyperparameter settings mentioned above, the following results were obtained:
<br>Best run id: <br>
Accuracy: <br>

### Model Comparison
For the task of predicting heart failure using a dataset from Kaggle, there is no doubt that the model trained using HyperDrive was more accurate than the model trained using AutoML. On top of that, the execution time was lower using HyperDrive than AutoML, since just one model was evaluated in the first approach compared with the multiple models tried by AutoML. In summary, despite the higher effort required in the former, it appears that HyperDrive could lead to better results whenever the model and the hyperparameter settings are chosen wisely beforehand.

## Model Deployment
Once both models were compared, the Hyperdrive model was chosen to be deployed as a web service. The following were the deployment steps:<br>
* A scoring script was defined.
* A new environment was created from conda dependencies previously specified.
* The scoring script was combined with the environment in inference configuration.
* A deployment configuration was set with 1 cpu core and 1 GB RAM.
* The model, inference and deployment configuration was specified, along with the name and deployment location of the web service.
* Once deployed, the scoring URI was obtained.
* Data from the endpoint.py and the scoring URI was used to query the endpoint.
* The logs of the service were obtained.

## Future improvements

The following are some suggestions to improve the model quality:<br>
* The dataset is highly imbalanced, so by aplying balancing techniques, such as resampling, cross-validation, clustering, and so on, could be ways of improving the model outcome.
* Data cleaning is always a great tool for improving model accuracy, specially as soon as the volume of data increases.
* Azure cloud resources for monitoring applications, like azure monitor or app insights, could be applied to track the performance of the web service. 

## Screen Recording



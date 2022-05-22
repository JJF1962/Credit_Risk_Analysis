# Overview of the analysis
The objective of this analysis is to identify a predictive model, that allows manage properly sensitivity (recall) and precision in the data, to predict credit risk with machine learning models, builded and evaluated using Python, please see as follows the tools and libraries used during of this module and challenge execution.

Summary of the tools, packages and libraries used: 
* Packages: Python 3.7 and Anaconda packages
* Computer Language: Python.
* The package dependencies are satisfied in the  mlenv environment: NumPy, version 1.11 or later, SciPy, version 0.17 or later and 
Scikit-learn, version 0.21 or later, obviously it was installed the imbalanced-learn Package.
* Tools: Jupyter Notebook, Colab and Git Hub, additionally it was used big range of libraries such as:
* Data: It was  used the LoanStats_2019Q1.csv dataset, and imbalanced-learn and scikit-learn libraries to perform the analysis. 
**  import warnings (warnings.filterwarnings('ignore'))
**  import numpy as np
**  import pandas as pd
**  from pathlib import Path
**  from collections import Counter
**  !pip install imblearn
**  from sklearn.ensemble import AdaBoostClassifier
**  from sklearn.metrics import balanced_accuracy_score
**  from sklearn.metrics import confusion_matrix
**  from imblearn.metrics import classification_report_imbalanced

It was learned  how to install and  set the proper enviroment melnv, machine learning algorithms used in data analytics, create training and test groups from a given data set, implement the logistic regression, decision tree, random forest, and support vector machine algorithms, interpret the results of the logistic regression, decision tree, random forest, compare the advantages and disadvantages of each supervised learning algorithm, determine which supervised learning algorithm is best used for a given data set or scenario and the use ensemble and resampling techniques to improve model performance.

The academic objective was to learn that Machine learning require the  use of statistical algorithms to perform tasks such as learning from data patterns and making predictions. There are many different models—a model is a mathematical representation of something that happens in the real world,Broadly speaking, machine learning can be divided into three learning categories: supervised, unsupervised, and deep. For the class purposes, it was focus only in supervised and unsupervised learning. Saying that we move to present the results.

# Results
The Challenge consisted in three technical analysis deliverables and a written report:

* Deliverable 1: Use Resampling Models to Predict Credit Risk
* Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
* Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
* Deliverable 4: A Written Report on the Credit Risk Analysis (README.md)

The following models were used to execute this challenge: 
* Naive Random Oversampling
* SMOTE Oversampling.
* Undersampling.
* SMOTEENN Sampling.
* Balanced Random Forest Classifying.
* Easy Ensemble Classifying.

For each Deliverable and model a report shows accurancy score, confusion matrix, and clasification report; the scores represent key indicators for the analisis of the balance between sensitivity and precision, allowing to identify and eliminate False Positive and False Negatives.
For the execution of the Delivery one and two, it was used Credit Risk Resampling Techniques that consisted in the following
## Deliverable 1 -2: Use Resampling Models to Predict Credit Risk 
### Deliverable 1
Consisted in to Read the CSV and Perform Basic Data Cleaning, split the Data into training and testing, oversampling and Naive Random Oversampling
The LoanStats_2019Q1.csv dataset was imported and cleaned using specific codes and methodology, we executed all  the code in  Jupyter Notebook originally named credit_risk_resampling starter-code.ipynb, re-named to credit_risk_resampling.ipynb 

#### Naive Random Oversampling

* See in the figure below the accurancy score equal to: 0.684

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery%201%20balaced%20accurancy%20score.%20PNG.PNG)

As shown in image below the other results for the Delivery 1 were:

* Presision (High Risk):  1%
* Precision (Low Risk): 100%
* Sensitivity - Recall (High Risk): 60%
* Sensitivity - Recall (High Risk): 68%

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery%201%20Confusion%20matrix%20and%20imbalance%20classification%20report.PNG)

### Delivery 2
#### SMOTE Oversampling

* Accuracy score: 64%
* Precision (High risk): 1%
* Precision (Low risk): 100%
* Recall (High risk): 60%
* Recall (low risk): 68%

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery2%20Smooteenn%20algorithm.PNG)


#### Undersampling

* Accuracy score: 53%
* Precision (High risk): 1%
* Precision (Low risk): 100%
* Recall (High risk): 61%
* Recall (low risk): 45%

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery2%20undersampling.PNG)

#### Combination (Over and Under) Sampling

* Accuracy score: 53%
* Precision (High risk): 1%
* Precision (Low risk): 100%
* Recall (High risk): 70%
* Recall (low risk): 58%

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery%202%20(4)%20Combination.PNG)


## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
### Deliver 3 

#### Balanced Random Forest Classifying

Accuracy score: 90%
Precision (High risk): 4%
Precision (Low risk): 100%
Recall (High risk): 65%
Recall (low risk): 90%

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Deliver%203%20Balance%20Random%20Forest%20Classifying.PNG) 

#### # Easy Ensemble AdaBoost Classifier

Accuracy score: 90%
Precision (High risk): 4%
Precision (Low risk): 100%
Recall (High risk): 65%
Recall (low risk): 90%

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery%203%20Easy%20Ensemble%20AdaBoost%20Classifier.PNG)

## Deliverable 4: A Written Report on the Credit Risk Analysis (README.md)

The Challenge consist in create a  report should contain the following:
* Overview of the analysis: Explain the purpose of this analysis.
* Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.
* Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

Starting to summarize The first step was in each model to evaluate and measure accuracy, trying to see results closed  1 (100%). As reviewed during this module the accuracy scores permit to narrow down the analysis, as a coseuencce that permit to  eliminate observations returning too many false returns. Specifically in the credit risk analysis, the percent obtained of low & high isk observations were  identified and are true. The following step was to identify precision, determining how precise predicted low & high risk evaluates to actual results. The last step consisted in to check sensivity (recall),ttempting to balance low & high risk predictors. 


# Summary
* It was a good learning experience, require to think different managing and evaluating data, it was very  intense and I need to continue and reviewing several concept that need more attention and learning from my side.
* Executing the delivery 3, it was very challenged for me, some codes no ran properly in the Jupyter Notboo in my specific case, derefor, I executed the Delivery 3 in Google Colab editor as shown in the image below:

![This is an image](https://github.com/JJF1962/Credit_Risk_Analysis)

* After evaluate the models, the Blanced Random Classifier and the Easy Ensemble shown a high accuracy rate of 90%, while, the  other accuracy scores were below 70%, which I consideredo low , if we are looking to predict credit risk. therefore,  the mentioned models are adquate for prediction of credit risk.


# Overview of the analysis
The objective of this analysis is to identify a predictive model, that allows manage properly sensitivity (recall) and precision in the data, to predict credit risk with machine learning models, builded and evaluated using Python, pleasee as follows the tools and libraries used during of this module and challenge execution.

Tools used: 
Packages: Python 3.7 and Anaconda packages
Computer Language: Python.
The package dependencies are satisfied in our mlenv environment: NumPy, version 1.11 or later, SciPy, version 0.17 or later and 
Scikit-learn, version 0.21 or later, obviously it was installed the imbalanced-learn Package.
Tools: Jupyter Notebook, Colab and Git Hub, additionally it was used big range of libraries such as:
Data: It was  used the LoanStats_2019Q1.csv dataset, and imbalanced-learn and scikit-learn libraries to perform the analysis. 
* import warnings (warnings.filterwarnings('ignore'))
* import numpy as np
* import pandas as pd
* from pathlib import Path
* from collections import Counter
* !pip install imblearn
* from sklearn.ensemble import AdaBoostClassifier
* from sklearn.metrics import balanced_accuracy_score
* from sklearn.metrics import confusion_matrix
* from imblearn.metrics import classification_report_imbalanced

We learned  how to install and  set the proper enviroment melnv, machine learning algorithms used in data analytics, create training and test groups from a given data set, implement the logistic regression, decision tree, random forest, and support vector machine algorithms, interpret the results of the logistic regression, decision tree, random forest, compare the advantages and disadvantages of each supervised learning algorithm, determine which supervised learning algorithm is best used for a given data set or scenario and the use ensemble and resampling techniques to improve model performance.

The academic objective was to learn that Machine learning require the  use of statistical algorithms to perform tasks such as learning from data patterns and making predictions. There are many different models—a model is a mathematical representation of something that happens in the real world,Broadly speaking, machine learning can be divided into three learning categories: supervised, unsupervised, and deep. For our purposes, we'll only discuss supervised and unsupervised learning. Saying that we move to present the results.

# Results
The Challenge consisted in consists of three technical analysis deliverables and a written report:

* Deliverable 1: Use Resampling Models to Predict Credit Risk
* Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
* Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
* Deliverable 4: A Written Report on the Credit Risk Analysis (README.md)

The following models will be used in the analysis: 
* Naive Random Oversampling
* SMOTE Oversampling.
* Cluster Centroid Undersampling.
* SMOTEENN Sampling.
* Balanced Random Forest Classifying.
* Easy Ensemble Classifying.

For each Deliverable and model a report shows accurancy score, confusion matrix, and clasification report; the scores represent key indicators for the analisis of the balance between sensitivity and precision, allowing to identify and eliminate False Positive and False Negatives.
For the execution of the Delivery one and two, it was used Credit Risk Resampling Techniques that consisted in the following
## Deliverable 1: Use Resampling Models to Predict Credit Risk 
Consisted in to Read the CSV and Perform Basic Data Cleaning, split the Data into training and testing, oversampling and Naive Random Oversampling
The LoanStats_2019Q1.csv dataset was imported and cleaned using specific codes and methodology, we executed all  the code in  Jupyter Notebook originally named credit_risk_resampling starter-code.ipynb, re-named to credit_risk_resampling.ipynb 
* See in the figure below the accurancy score equal to: 0.684

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery%201%20balaced%20accurancy%20score.%20PNG.PNG)

As shown in image below the other results for the Delivery 1 were:

* Presision (High Risk):  1%
* Precision (Low Risk): 100%
* Sensitivity - Recall (High Risk): 60%
* Sensitivity - Recall (High Risk): 68%

![insert an image](https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery%201%20Confusion%20matrix%20and%20imbalance%20classification%20report.PNG)

## Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk

![insert an image (https://github.com/JJF1962/Credit_Risk_Analysis/blob/main/Images/Delivery%202%20%20(3)%20Combination%20%20balance%20acc%20score%2C%20conf%20matrix%2C%20imbalance%20class%20report.PNG)

The results for Delivery 2 - SMOTE Oversampling, were:
Accuracy score: 66%
Precision (High risk): 1%
Precision (Low risk): 100%
Recall (High risk): 62%
Recall (low risk): 69%

## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk





## Deliverable 4: A Written Report on the Credit Risk Analysis (README.md)




# Summary

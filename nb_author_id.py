#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB


print("All Import Statements Are Working")
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#Create Naive Bayes Classifier 
emailNB = GaussianNB()

#Train the classifier on the training features and labels
t0 = time()
emailNB.fit(features_train, labels_train)
print("Training Time", round(time()-t0,3), "seconds")

#Predict labels 
t1 = time()
emailNB.predict(features_test)
print("Prediction Time", round(time()-t1,3), "seconds")

print("The accuracy of the Naive Bayes Classifier is ", emailNB.score(features_test,labels_test) )



#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


from sklearn import tree

# A decision tree to identify the author of the emails between Sara and Chris

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

t0 = time()
decisionTreeAuthor = tree.DecisionTreeClassifier(min_samples_split = 40)
decisionTreeAuthor = decisionTreeAuthor.fit(features_train, labels_train)
print("Training Time for Decision Tree", round(time()-t0,3), "seconds")

t1 = time()
decisionTreeAuthor.predict(features_test)
print("Prediction Time for Decision Tree", round(time()-t1,3), "seconds")


print("The accuracy of the decision tree is ", decisionTreeAuthor.score(features_test, labels_test))

print("The number of features is ", len(features_train[0]))


#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#features_train = features_train[:len(features_train)//100]
#labels_train = labels_train[:len(labels_train)//100]


print("All import loaded")
t0 = time()
authorSVM = svm.SVC(C = 10000.0, kernel = 'rbf')

print(" SVC Created")
authorSVM.fit(features_train, labels_train)
print("Training Time for Linear SVM", round(time()-t0,3), "seconds")

t1 = time()
prediction =authorSVM.predict(features_test)
print("Prediction Time for Linear SVM", round(time()-t1,3), "seconds")

print("The accuracy of the Linear SVM is ", authorSVM.score(features_test, labels_test))

print("The predicted label for the 10th element is ", prediction[10])

print("The predicted label for the 26th element is ", prediction[26])

print("The predicted label for the 50th element is ", prediction[50])
print(len(prediction))

sum = 0
for i in prediction:
	if i == 1:
		sum = sum + 1

print("The total number of predicted events to tbe in the Chris class are ", sum)



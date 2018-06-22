#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
from sklearn import tree
import graphviz

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "C:/PythonFiles/ud120-projects-master/text_learning/your_word_data.pkl" 
authors_file = "C:/PythonFiles/ud120-projects-master/text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )




### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier

#from sklearn import cross_validation
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

#For Scikit-learn version 0.19
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()



### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
treeEmail  = tree.DecisionTreeClassifier()
treeEmail = treeEmail.fit(features_train,labels_train)

#tree_data = tree.export_graphviz(treeEmail,out_file = None)
#graph = graphviz.Source(tree_data)
#graph.render("emailTree")
print("The accuracy of the Decision Tree is ", treeEmail.score(features_test, labels_test))

featureArray = treeEmail.feature_importances_

wordsList = vectorizer.get_feature_names()
for i in range(0,len(featureArray)):
	if featureArray[i] >= 0.2:
		print("The feature number is %d with importance %.2f" % (i,featureArray[i]))
		print("The word with the problem is " + wordsList[i])

#wordsList = vectorizer.get_feature_names()
print("The length of the word features is ", len(wordsList))
print("The length of the featureArray is ", len(featureArray))
maxWord = max(featureArray)




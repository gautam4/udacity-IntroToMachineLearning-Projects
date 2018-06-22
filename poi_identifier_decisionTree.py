#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

#Project: Enron Email Data Analysis Decision Tree 
#Completed as part of the final project for Udacity's Intro to Machine Learning Course
#Modified the starter code(Import form dataset file) from the poi_id.py to import feature datasets

#Goal: To build a person of interest identifier using the Enron individual income and email dataset

#Approach: 1. Apply a Naive Bayes classifier to each individual feature to identify the most relevant 
#				- Six features with a Naive Bayes score accuracy below 0.6 were excluded from further analysis
#				- Two Features with a Naive Bayes score accuracy of 1.0 were excluded from further analysis
# 		   2. A decision tree classifer was constructed using the remaining features
#		   3. The decision tree classifier was trained on the poi_train training sample and tested on an independent poi_test sample
#		   4. The accuracy of the decision tree classifer was 0.8636


#Open Project dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

#Dataset for the Decision Tree
features_list = ['poi','deferral_payments', 'total_payments','restricted_stock_deferred','deferred_income',
				'total_stock_value','exercised_stock_options','restricted_stock','director_fees','from_poi_to_this_person', 'from_this_person_to_poi'] 
features_data = featureFormat(my_dataset, features_list, sort_keys = True)
poi_labels, poi_features = targetFeatureSplit(features_data)

#Datasets for the Naive Bayes Classifiers

#Six Features Excluded because of Naive Bayes Score Accuracy being below 0.6

salary_list = ['poi','salary']
salary_data = featureFormat(my_dataset, salary_list, sort_keys = True)
salary_labels, salary_features = targetFeatureSplit(salary_data)

expenses_list = ['poi','expenses']
expenses_data = featureFormat(my_dataset, expenses_list, sort_keys = True)
expenses_labels, expenses_features = targetFeatureSplit(expenses_data)

long_term_list = ['poi','long_term_incentive']
long_term_data = featureFormat(my_dataset, long_term_list, sort_keys = True)
long_term_labels, long_term_features = targetFeatureSplit(long_term_data)

bonus_list = ['poi','bonus']
bonus_data = featureFormat(my_dataset, bonus_list, sort_keys = True)
bonus_labels, bonus_features = targetFeatureSplit(bonus_data)

loan_list = ['poi','loan_advances']
loan_data = featureFormat(my_dataset, loan_list, sort_keys = True)
loan_labels, loan_features = targetFeatureSplit(loan_data)

other_list = ['poi','other']
other_data = featureFormat(my_dataset, other_list, sort_keys = True)
other_labels, other_features = targetFeatureSplit(other_data)

#Two Features Excluded because of Naive Bayes Score Accuracy of 1.0
director_stock = ['poi','director_fees']
director_stock_data = featureFormat(my_dataset, director_stock, sort_keys = True)
director_stock_labels, director_stock_features = targetFeatureSplit(director_stock_data)

stock_list = ['poi','restricted_stock_deferred']
stock_data = featureFormat(my_dataset, stock_list, sort_keys = True)
stock_labels, stock_features = targetFeatureSplit(stock_data)

deferal_list = ['poi','deferral_payments']
deferal_data = featureFormat(my_dataset, deferal_list, sort_keys = True)
deferal_labels, deferal_features = targetFeatureSplit(deferal_data)

total_list = ['poi','total_payments']
total_data = featureFormat(my_dataset, total_list, sort_keys = True)
total_labels, total_features = targetFeatureSplit(total_data)

deferred_list = ['poi','deferred_income']
deferred_data = featureFormat(my_dataset, deferred_list, sort_keys = True)
deferred_labels, deferred_features = targetFeatureSplit(deferred_data)

total_stock_list = ['poi','total_stock_value']
total_stock_data = featureFormat(my_dataset, total_stock_list, sort_keys = True)
total_stock_labels, total_stock_features = targetFeatureSplit(total_stock_data)

exercise_stock_list = ['poi','exercised_stock_options']
exercise_stock_data = featureFormat(my_dataset, exercise_stock_list, sort_keys = True)
exercise_stock_labels, exercise_stock_features = targetFeatureSplit(exercise_stock_data)

restricted_stock = ['poi','restricted_stock']
restricted_stock_data = featureFormat(my_dataset, restricted_stock, sort_keys = True)
restricted_stock_labels, restricted_stock_features = targetFeatureSplit(restricted_stock_data)

fromPOIPerson_list = ['poi','from_poi_to_this_person']
fromPOIPerson_data = featureFormat(my_dataset, fromPOIPerson_list, sort_keys = True)
fromPOIPerson_labels, fromPOIPerson_features = targetFeatureSplit(fromPOIPerson_data)

toPOIPerson_list = ['poi','from_this_person_to_poi']
toPOIPerson_data = featureFormat(my_dataset, toPOIPerson_list, sort_keys = True)
toPOIPerson_labels, toPOIPerson_features = targetFeatureSplit(toPOIPerson_data)

#Split into Different Dataset

#Training and Testing Sample Split for Naive Bayes Classifiers
deferal_train, deferal_test, deferal_labels_train, deferal_labels_test = train_test_split(deferal_features, deferal_labels, test_size=0.3, random_state=42)
total_train, total_test, total_labels_train, total_labels_test = train_test_split(total_features, total_labels, test_size=0.3, random_state=42)
deferred_train, deferred_test, deferred_labels_train, deferred_labels_test = train_test_split(deferred_features, deferred_labels, test_size=0.3, random_state=42)
total_stock_train, total_stock_test, total_stock_labels_train, total_stock_labels_test = train_test_split(total_stock_features, total_stock_labels, test_size=0.3, random_state=42)
exercise_stock_train, exercise_stock_test, exercise_stock_labels_train, exercise_stock_labels_test = train_test_split(exercise_stock_features, exercise_stock_labels, test_size=0.3, random_state=42)
restricted_stock_train, restricted_stock_test, restricted_stock_labels_train, restricted_stock_labels_test = train_test_split(restricted_stock_features, restricted_stock_labels, test_size=0.3, random_state=42)
fromPOIPerson_train, fromPOIPerson_test, fromPOIPerson_labels_train, fromPOIPerson_labels_test = train_test_split(fromPOIPerson_features, fromPOIPerson_labels, test_size=0.3, random_state=42)
toPOIPerson_train, toPOIPerson_test, toPOIPerson_labels_train, toPOIPerson_labels_test = train_test_split(toPOIPerson_features, toPOIPerson_labels, test_size=0.3, random_state=42)

#Six Features Excluded from decision tree because of Naive Bayes Score Accuracy being below 0.6
salary_train, salary_test, salary_labels_train, salary_labels_test = train_test_split(salary_features, salary_labels, test_size=0.3, random_state=42)
expenses_train, expenses_test, expenses_labels_train, expenses_labels_test = train_test_split(expenses_features, expenses_labels, test_size=0.3, random_state=42)
long_term_train, long_term_test, long_term_labels_train, long_term_labels_test = train_test_split(long_term_features, long_term_labels, test_size=0.3, random_state=42)
bonus_train, bonus_test, bonus_labels_train, bonus_labels_test = train_test_split(bonus_features, bonus_labels, test_size=0.3, random_state=42)
loan_train, loan_test, loan_labels_train, loan_labels_test = train_test_split(loan_features, loan_labels, test_size=0.3, random_state=42)
other_train, other_test, other_labels_train, other_labels_test = train_test_split(other_features, other_labels, test_size=0.3, random_state=42)

#Two features excluded from decision tree because of Naive Bayes Score Accuracy of 1.0
stock_train, stock_test, stock_labels_train, stock_labels_test = train_test_split(stock_features, stock_labels, test_size=0.4, random_state=42)
director_stock_train, director_stock_test, director_stock_labels_train, director_stock_labels_test = train_test_split(director_stock_features, director_stock_labels, test_size=0.4, random_state=42)

#Training and Testing Sample Split for Decision Tree Classifiers
poi_train, poi_test, poi_labels_train, poi_labels_test = train_test_split(poi_features, poi_labels, test_size = 0.3, random_state=42)

#Classifiers for each trait

#Naive bayes classifiers run on all features
deferal = GaussianNB()
deferal.fit(deferal_train, deferal_labels_train)
print("The score of the naive bayes with Deferal and Poi is ", deferal.score(deferal_test, deferal_labels_test))

total = GaussianNB()
total.fit(total_train, total_labels_train)
print("The score of the naive bayes with Total and Poi is ", total.score(total_test, total_labels_test))

deferred = GaussianNB()
deferred.fit(deferred_train, deferred_labels_train)
print("The score of the naive bayes with Deferred and Poi is ", deferred.score(deferred_test, deferred_labels_test))

totalStock = GaussianNB()
totalStock.fit(total_stock_train, total_stock_labels_train)
print("The score of the naive bayes with Total Stock and Poi is ", totalStock.score(total_stock_test, total_stock_labels_test))

exerciseStock = GaussianNB()
exerciseStock.fit(exercise_stock_train, exercise_stock_labels_train)
print("The score of the naive bayes with exercise stock and Poi is ", exerciseStock.score(exercise_stock_test, exercise_stock_labels_test))

restrictedStock = GaussianNB()
restrictedStock.fit(restricted_stock_train, restricted_stock_labels_train)
print("The score of the naive bayes with restricted stock and Poi is ", restrictedStock.score(restricted_stock_test, restricted_stock_labels_test))

fromPOI = GaussianNB()
fromPOI.fit(fromPOIPerson_train, fromPOIPerson_labels_train)
print("The score of the naive bayes with fromPOI and Poi is ", fromPOI.score(fromPOIPerson_test, fromPOIPerson_labels_test))

toPOI = GaussianNB()
toPOI.fit(toPOIPerson_train, toPOIPerson_labels_train)
print("The score of the naive bayes with toPOI and Poi is ", toPOI.score(toPOIPerson_test, toPOIPerson_labels_test))

#Six Features Excluded from decision tree because of Naive Bayes Score Accuracy being below 0.6
salary = GaussianNB()
salary.fit(salary_train,salary_labels_train)
print("The score of the naive bayes with Salary and Poi is ", salary.score(salary_test, salary_labels_test))

expenses = GaussianNB()
expenses.fit(expenses_train, expenses_labels_train)
print("The score of the naive bayes with expenses and Poi is ", expenses.score(expenses_test, expenses_labels_test))

longTerm = GaussianNB()
longTerm.fit(long_term_train, long_term_labels_train)
print("The score of the naive bayes with long term and Poi is ", longTerm.score(long_term_test, long_term_labels_test))

loan = GaussianNB()
loan.fit(loan_train, loan_labels_train)
print("The score of the naive bayes with Loan and Poi is ", loan.score(loan_test, loan_labels_test))

bonus = GaussianNB()
bonus.fit(bonus_train, bonus_labels_train)
print("The score of the naive bayes with Bonus and Poi is ", bonus.score(bonus_test, bonus_labels_test))

other = GaussianNB()
other.fit(other_train, other_labels_train)
print("The score of the naive bayes with other and Poi is ", other.score(exercise_stock_test, exercise_stock_labels_test))

#Two Features excluded from decision tree because of Naive Bayes Score of 1.0
stock = GaussianNB()
stock.fit(stock_train, stock_labels_train)
print("The score of the naive bayes with Stock and Poi is ", stock.score(stock_test, stock_labels_test))

directorStock = GaussianNB()
directorStock.fit(director_stock_train, director_stock_labels_train)
print("The score of the naive bayes with director stock and Poi is ", directorStock.score(director_stock_test, director_stock_labels_test))

#Classifier: Use a Decision Tree with the remaining features
poiTree = tree.DecisionTreeClassifier(min_samples_split = 40)
poiTree = poiTree.fit(poi_train, poi_labels_train)

print("The accuracy of the decision tree is ", poiTree.score(poi_test, poi_labels_test))
print("The importance of features is :")
print(poiTree.feature_importances_)



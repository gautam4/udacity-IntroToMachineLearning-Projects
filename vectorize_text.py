#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        #if temp_counter < 200:
        path = os.path.join('C:/Datasets/enron_mail_20150507/enron_mail/', path[:-2])
        path = path + "_"
        email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
        email_words = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
        email_words = email_words.replace("sara", "")
        email_words = email_words.replace("shackleton", "")
        email_words = email_words.replace("chris", "")
        email_words = email_words.replace("germani", "")

        #Removed the word that was found to have the greatest decision ability because it is related to Sara's name 
        email_words = email_words.replace("sshacklensf","")
        email_words = email_words.replace("cgermannsf", "")
            ### append the text to word_data
        word_data.append(email_words)
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if from_person is from_sara:
                #print("The person is Identified as sara but is. ", from_person)
            from_data.append(0)
        elif from_person is from_chris:
                #print("The person is Identified as chris but is  ", from_person)
            from_data.append(1)



        email.close()

print ("emails processed")
from_sara.close()
from_chris.close()

#print(from_data)

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )

#print("The word data at index 152 is ", word_data[152])



#print("The number of elements in word data is ", len(word_data))
#print(word_data)

### in Part 4, do TfIdf vectorization here

#vectorizer = CountVectorizer(stop_words = 'english', analyzer = 'word')
#print("Vectorizer is created")
#vector = vectorizer.fit_transform(word_data)
#print(vector.shape)

#tf_transformer = TfidfTransformer()
#train_tfidf = tf_transformer.fit_transform( vector)
#print("The shape of the tf IDF vector is ", vector.shape)
#print("The number of words based on count vectorizer is ", len(vectorizer.get_feature_names()))
#arr = vectorizer.get_feature_names()
#print("The number of words based on count vectorizer is ", arr[34597])

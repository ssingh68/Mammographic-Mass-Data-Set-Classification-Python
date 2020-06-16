# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:24:49 2019

@author: shrey
"""

#Assignment 5 - Option 2 (CS504)

#Importing Libraries
import numpy as np 
import pandas as pd 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#Read csv
Mammo = pd.read_csv("C:/Users/shrey/Desktop/George Mason University/Sem 2/CS 504/Assignment/mammographic.csv", sep=',', 
                  names = ["BI-RADS assessment", "Age", "Shape", "Margin","Density","Severity"])
Mammo.info()
Mammo.head()

#Data Preprocessing
#Replace '?' with Empty String
Mammo = Mammo.replace('?',None)

#Replace Age Attribute Empty Strings by Mean of all Age values
Mammo['Age'].fillna(Mammo['Age'].mean(), inplace=True)

#Replace empty strings with NAN
Mammo.replace({"":np.nan}, inplace=True)

#Replace all NAN values by mode
Mammo = Mammo.fillna(Mammo.mode().iloc[0])

#Binning done for Age attributes and created a new column named Age Binned
bins = [0,20,40,60,80,100]
labels = [1,2,3,4,5]
Mammo['Age Binned'] = pd.cut(Mammo['Age'].astype(int), bins,labels=labels)

#Removing BI-RADS assessment as not a predicitve attribute
Mammo = Mammo.drop(['BI-RADS assessment'],axis =1)


#Building Models
#x -> All attributes except the Predicting Variable
x = Mammo.iloc[:,[0,1,2,3,5]]

#Assign Ordinal Levels to Categorical data using LabelEncoder
le = LabelEncoder()

#y -> The Prediciting Variable Severity
y = le.fit_transform(Mammo['Severity'])

#1. a) Decision Tree Model using Test Train Split
X_train, X_test, y_train, y_test = train_test_split (x, y, test_size=0.25, random_state=100)
Classifier = DecisionTreeClassifier()
Classifier = Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
#Calculating Accuracy of the model
print("Accuracy for Decision Tree using Test-Train Split CV:",metrics.accuracy_score(y_test, y_pred))

#1. b) Decision Tree Model using K fold CV
clf = tree.DecisionTreeClassifier()
#Calculating Accuracy of the model
Accu1=cross_val_score(clf, x, y, cv=10)
print("Accuracy for Decision Tree using K Fold CV:",np.mean(Accu1))


#2. Random Forest Model using K fold CV 
# Instantiate the grid search model
# Fit the grid search to the data
rf = RandomForestClassifier(
         bootstrap= True,
         max_depth= 3,
         max_features= 2,
         min_samples_leaf= 1,
         min_samples_split= 11,
         n_estimators= 10,
         random_state=5)
#Calculating Accuracy of the model
Accu2=cross_val_score(rf, x, y, cv=10)
print("Accuracy for Random Forest using using K Fold CV:",np.mean(Accu2))

#3. KNN Model using K fold CV
#Running for loop to check the best accuracy score
#for i in range(1,51):
#    classifier = KNeighborsClassifier(n_neighbors=i) 
#    Accu3=cross_val_score(classifier, x, y, cv=10)
#    print(np.mean(Accu3))

#Best Score - Calculating Accuracy of the model
classifier = KNeighborsClassifier(n_neighbors=13) 
Accu3=cross_val_score(classifier, x, y, cv=10)
print("Accuracy for KNN using using K Fold CV:", np.mean(Accu3))

#4. Naive Bayes Model using K fold CV

nb = MultinomialNB()
#Calculating Accuracy of the model
Accu4=cross_val_score(nb, x, y, cv=10)
print("Accuracy for Naive Bayes using using K Fold CV:", np.mean(Accu4))

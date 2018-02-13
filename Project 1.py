# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:04:15 2018

@author: HP
"""
# Code copied from 
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

'''
First of all, to check the versions of all the libraries that are needed to run the classifiers,
run the following LOC and see if they are not older than the versions mentioned below

Python: 2.7.11 (default, Mar  1 2016, 18:40:10) 
[GCC 4.2.1 Compatible Apple LLVM 7.0.2 (clang-700.1.81)]
scipy: 0.17.0
numpy: 1.10.4
matplotlib: 1.5.1
pandas: 0.17.1
sklearn: 0.18.1

'''
# python
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# Loading the libraries
import pandas
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Loading the IRIS dataset from the UCI repository
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
filename = 'iris.csv'
names= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = pandas.read_csv(filename, names=names)


#Dimensions of dataset
print(dataset.shape)

#Peek of the data
print(dataset.head(10))

#Summary of dataset
print(dataset.describe())

#Class distribution
print(dataset.groupby('class').size())



print('Hogyaaaa')




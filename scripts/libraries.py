'''
Python modules required for running the analyses
All modules run in Python 3.9.7
'''

#%% LIBRARIES

#   Data manipulation
import numpy as np
import pandas as pd
import copy

#   Statistics
from numpy import mean
from numpy import std
from numpy import sqrt

#   Multivariate imputation 
from sklearn.experimental import enable_iterative_imputer # Dynamically sets IterativeImputer as an attribute of the impute module
from sklearn.impute import IterativeImputer

#   Data visualisation
import seaborn as sb

#   Feature selection

from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier

#   Principal components analysis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#   ML methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#   Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.metrics import confusion_matrix

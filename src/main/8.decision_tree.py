# A decision tree was the second best model

# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules
from features.build_features import *

# Public modules
import seaborn as sns
import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, \
                            precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from numpy.random import seed
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
                            f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
                                          QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

# Inputs
SHOW_ERROR_ANALYSIS = True

# Extract
seed(40)
train = read_csv("../../data/interim/train.csv")
train_y = train[["churn"]].values
dev = read_csv("../../data/interim/dev.csv")
dev_y = dev[["churn"]].values

# Data parameters
features_pipeline = data_preparation()

# Model parameters
full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", lgb.LGBMClassifier()),
])

clf = DecisionTreeClassifier(max_depth=5)

# Evaluate each model
scores = []
names = []

full_pipeline = Pipeline([
    ("features", features_pipeline),
    ("clf", clf),
])
full_pipeline.fit(train, train_y)
pred_y = full_pipeline.predict(dev)
score = accuracy_score(dev_y, pred_y)
print("Accuracy: %.3f" % (score))

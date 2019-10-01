# A decision tree depth 5 did well

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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
                            f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
                                          QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import graphviz
from sklearn import tree
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
temp = train.drop(["churn"], axis=1).fillna(0)
X_column_names = temp.columns
train_X = temp.values
train_y = train[["churn"]].values
dev = read_csv("../../data/interim/dev.csv")
dev_X = dev.drop(["churn"], axis=1).fillna(0).values
dev_y = dev[["churn"]].values


# Data parameters
features_pipeline = data_preparation()

clf = DecisionTreeClassifier(max_depth=5)

clf.fit(train_X, train_y)
pred_y = clf.predict(dev_X)
score = accuracy_score(dev_y, pred_y)
print("Accuracy: %.3f" % (score))

dot_data = tree.export_graphviz(clf, out_file=None,
                      feature_names=train.drop(["churn"], axis=1).columns,
                      class_names=["retained", "churned"],
                      filled=True, rounded=True,
                      special_characters=True)
graph = graphviz.Source(dot_data)
graph.render()

os.remove("Source.gv")
os.rename("Source.gv.pdf", "../../reports/main_outputs/8.decision_tree.pdf")

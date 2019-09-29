# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules
from features.build_features import *

# Public modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, \
                            precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import lightgbm as lgb

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
    # ("clf", LogisticRegression(random_state=1)),
    ("clf", lgb.LGBMClassifier()),
])

# Learning curve
train_sizes = (np.linspace(0.1, 1.0, 10)*train.shape[0]).astype(int)
training_scores, dev_scores = [], []
for train_size in train_sizes:
    temp_train = train.iloc[0:train_size]
    temp_train_y = temp_train[["churn"]].values
    full_pipeline.fit(temp_train, temp_train_y)
    train_pred_y = full_pipeline.predict(train)
    dev_pred_y = full_pipeline.predict(dev)
    training_scores.append(accuracy_score(train_y, train_pred_y))
    dev_scores.append(accuracy_score(dev_y, dev_pred_y))


plt.plot(train_sizes, training_scores,
         color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, dev_scores,
         color='green', marker='o', markersize=5, label='dev accuracy')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Recall')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.0])
plt.savefig("../../reports/main_outputs/5.lgbm_learning_curve.png")
plt.show()

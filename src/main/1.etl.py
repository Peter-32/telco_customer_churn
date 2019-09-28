# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules


# Public modules
import re
from numpy.random import seed
from pandas import read_csv
from sklearn.model_selection import train_test_split

# Set seed - ensures that the datasets are split the same way if re-run
seed(40)

# Extract
df = read_csv("../../data/raw/bank-additional-full.csv", sep=";")

# Transform target to numeric
df['Churn'] = df['Churn'].apply(lambda x: 1 if x.lower().strip() == "Yes" else 0)

# This column isn't needed to build our model on train/dev/test
df.drop(["customerID"], axis=1, inplace=True)

# Change columns to snake case
def camel_to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
df.columns = [camel_to_snake_case(x) for x in df.columns]

# Split datasets
train, temp = train_test_split(df, test_size=0.40)
dev, test = train_test_split(temp, test_size=0.50)

# Write results to files
train.to_csv("../../data/interim/train.csv", index=False)
dev.to_csv("../../data/interim/dev.csv", index=False)
test.to_csv("../../data/interim/test.csv", index=False)

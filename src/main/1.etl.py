# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# My modules


# Public modules
import re
import pandas as pd
from pandas import read_csv
from numpy.random import seed
from sklearn.model_selection import train_test_split

# Set seed - ensures that the datasets are split the same way if re-run
seed(32)

# Extract
df = read_csv("../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Transform target to numeric
df['Churn'] = df['Churn'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)

# This column isn't needed to build our model on train/dev/test
df.drop(["customerID"], axis=1, inplace=True)

# Change columns to snake case
def camel_to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
df.columns = [camel_to_snake_case(x) for x in df.columns]

# Fix total_charges
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

# Convert some binary categories to numeric
df['is_female'] = df['gender'].apply(lambda x: 1 if x.lower().strip() == "female" else 0)
df['has_partner'] = df['partner'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['has_dependents'] = df['dependents'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['has_phone_service'] = df['phone_service'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['multiple_lines__yes'] = df['multiple_lines'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['multiple_lines__no_phone_service'] = df['multiple_lines'].apply(lambda x: 1 if x.lower().strip() == "no phone service" else 0)
df['internet_service__fiber_optic'] = df['internet_service'].apply(lambda x: 1 if x.lower().strip() == "fiber optic" else 0)
df['internet_service__dsl'] = df['internet_service'].apply(lambda x: 1 if x.lower().strip() == "dsl" else 0)
df['online_security__yes'] = df['online_security'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['online_security__no_internet_service'] = df['online_security'].apply(lambda x: 1 if x.lower().strip() == "no internet service" else 0)
df['online_backup__yes'] = df['online_backup'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['online_backup__no_internet_service'] = df['online_backup'].apply(lambda x: 1 if x.lower().strip() == "no internet service" else 0)
df['device_protection__yes'] = df['device_protection'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['device_protection__no_internet_service'] = df['device_protection'].apply(lambda x: 1 if x.lower().strip() == "no internet service" else 0)
df['tech_support__yes'] = df['tech_support'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['tech_support__no_internet_service'] = df['tech_support'].apply(lambda x: 1 if x.lower().strip() == "no internet service" else 0)
df['streaming_tv__yes'] = df['streaming_tv'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['streaming_tv__no_internet_service'] = df['streaming_tv'].apply(lambda x: 1 if x.lower().strip() == "no internet service" else 0)
df['streaming_movies__yes'] = df['streaming_movies'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['streaming_movies__no_internet_service'] = df['streaming_movies'].apply(lambda x: 1 if x.lower().strip() == "no internet service" else 0)
df['contract__month_to_month'] = df['contract'].apply(lambda x: 1 if x.lower().strip() == "month-to-month" else 0)
df['contract__two_year'] = df['contract'].apply(lambda x: 1 if x.lower().strip() == "two year" else 0)
df['has_paperless_billing'] = df['paperless_billing'].apply(lambda x: 1 if x.lower().strip() == "yes" else 0)
df['payment_method__electronic_check'] = df['payment_method'].apply(lambda x: 1 if x.lower().strip() == "electronic check" else 0)
df['payment_method__mailed_check'] = df['payment_method'].apply(lambda x: 1 if x.lower().strip() == "mailed check" else 0)
df['payment_method__bank_transfer_automatic'] = df['payment_method'].apply(lambda x: 1 if x.lower().strip() == "bank transfer (automatic)" else 0)






df.drop(["gender", "partner", "dependents", "phone_service", "multiple_lines"], axis=1, inplace=True)

# Split datasets
train, temp = train_test_split(df, test_size=0.40, stratify=df['churn'])
dev, test = train_test_split(temp, test_size=0.50, stratify=temp['churn'])

# Write results to files
train.to_csv("../../data/interim/train.csv", index=False)
dev.to_csv("../../data/interim/dev.csv", index=False)
test.to_csv("../../data/interim/test.csv", index=False)

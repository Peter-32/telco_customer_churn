import numpy as np
from pandas import read_csv
from numpy.random import seed
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")

# Not used in this project
def _run_feature_selection(train):
    import shap
    import pandas as pd
    from numpy import cumsum
    from xgboost import XGBClassifier

    seed(40)

    train.fillna(0, inplace=True)

    # X and y
    X = train.drop(["churn"], axis=1)
    y = train[["churn"]]

    # lightgbm for large number of columns
    # import lightgbm as lgb; clf = lgb.LGBMClassifier()

    # Fit xgboost
    clf = XGBClassifier()
    clf.fit(X, y)

    # shap values
    shap_values = shap.TreeExplainer(clf).shap_values(X[0:10000])

    sorted_feature_importance = pd.DataFrame(shap_values, columns=X.columns).abs().sum().sort_values(ascending=False)
    cumulative_sum = cumsum([y for (x,y) in sorted_feature_importance.reset_index().values])
    gt_999_importance = cumulative_sum / cumulative_sum[-1] > .999
    nth_feature = min([y for (x,y) in zip(gt_999_importance, zip(range(len(gt_999_importance)))) if x])[0]
    important_columns = sorted_feature_importance.iloc[0:nth_feature+1].index.values.tolist()
    important_columns
    return important_columns


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

def data_preparation():
    # Extract
    train = read_csv("../../data/interim/train.csv", nrows=250)

    # Get column names by datatype
    binary_attribs = [col for col in train if np.isin(train[col].unique(), [0, 1]).all() if col != "churn"]
    num_attribs = train.drop(["churn"] + list(binary_attribs), axis=1).select_dtypes(exclude="O").columns

    # Numeric pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('std_scaler', StandardScaler()),
        ('minmax_scaler', MinMaxScaler()),
    ])

    # Binary pipeline
    binary_pipeline = Pipeline([
        ('selector', DataFrameSelector(binary_attribs)),
    ])

    # Combine pipelines
    features_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("binary_pipeline", binary_pipeline),
    ])

    # Return pipeline
    return features_pipeline

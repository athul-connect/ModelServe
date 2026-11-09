import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(x):

    numerical_columns = x.select_dtypes(
        include=["int64","float64"]
    ).columns.tolist()

    categorical_columns=x.select_dtypes(
        include=["object","category"]
    ).columns.tolist()

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",numerical_transformer,numerical_columns),
            ("cat",categorical_transformer,categorical_columns)
        ]
    )

    return preprocessor




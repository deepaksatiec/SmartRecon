import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope='module')
def data():
    return pd.read_csv('data/data_test.csv')


@pytest.fixture(scope='module')
def feat_types(data):
    def column_type(column: pd.Series):
        if not np.issubdtype(column.dtype, np.number):
            return 'categorical'
        else:
            return 'other'

    return {column: column_type(data[column]) for column in data.columns}

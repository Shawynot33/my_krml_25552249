import pandas as pd
from my_krml_25552249.features.impute import impute_missing

def test_impute_median():
    data = pd.Series([1, None, 3, None, 5])
    result = impute_missing(data, 'median')
    assert result.isnull().sum() == 0
    assert result[1] == 3  # median of [1,3,5] is 3

def test_impute_mean():
    data = pd.Series([1, None, 3, None, 5])
    result = impute_missing(data, 'mean')
    assert result.isnull().sum() == 0
    assert abs(result[1] - 3) < 1e-6  # mean of [1,3,5] is 3

def test_impute_mode():
    data = pd.Series([1, 1, 2, None, 3])
    result = impute_missing(data, 'mode')
    assert result.isnull().sum() == 0
    assert result[3] == 1  # mode is 1

def test_invalid_strategy():
    import pytest
    data = pd.Series([1, 2, None])
    with pytest.raises(ValueError):
        impute_missing(data, 'invalid')

from hashlib import sha1
import pandas as pd
import pytest
import sys
from decimal import Decimal
import numpy as np
import inspect


def test_1a(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert answer.size == 3, "Your array has the incorrect size. Are you slicing properly?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "3e1a4e12e3ba7129bcd27422d6a77dbb57ec68de", "Your array values are incorrect. Are you slicing properly?"
    return("Success")

def test_1b(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert 'numpy.ndarray' in str(type(answer)), "Your object is not of type 'numpy.ndarray'. Are you using the 'np.full()' function?"
    assert answer.shape == (2,2), "Your array dimensions are incorrect. Make sure you are creating a '2x2' array"
    assert str(answer.dtype) == 'float64', "Make sure your array is filled with floating point values."
    assert np.count_nonzero(answer == 3.4) == 4, "Your array values are incorrect. Make sure you are filling it with 3.4"
    return("Success")

def test_1c(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert 'numpy.ndarray' in str(type(answer)), "Your object is not of type 'numpy.arrary'. Are you using the 'np.full()' function?"
    assert answer.shape == (2, 3, 4), "Your array dimensions are incorrect. Make sure you are creating a '2x3X4' array"
    assert str(answer.dtype) == 'float64', "Make sure your array is filled with floating point values."
    assert np.count_nonzero(answer == 1) == 24.0, "Your array values are incorrect. Make sure you are using the 'np.ones()' function"
    return("Success")

def test_2a(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert answer.shape == (569, 20), "Your dataframe dimensions are incorrect. Are you reading in the correct dataframe?"
    assert str(answer['air_date'].dtype) == 'datetime64[ns]', "The 'air_date' column is of the incorrect type. Are you parsing it as dates?"
    return("Success")

def test_2c(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert answer.shape == (568,), "Your object dimensions are incorrect. The number of intervals should be 568. You may need to omit that NaT value we discussed."
    assert str(answer.dtype) == 'timedelta64[ns]', "Your object is of the incorrect data type. Are you parsing it as dates?"
    assert str(sum(answer.values)) == '364089600000000000 nanoseconds', "Some values in your object are incorrect."
    return("Success")

def test_2d(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert sha1(str(round(answer,2)).encode('utf8')).hexdigest() == "bc0d5950400d4df7183f1f23d44ed27def3095df", "Your answer is incorrect. Please try again. You may want to use a 'for' loop for this."
    return("Success")

def test_2e(answer):
    assert not answer is None, "Your object does not exist. Have you named your dataframe correctly?"
    assert str(answer['weekday_aired'].dtype) == 'object', "The 'weekday_aired' colum is of the incorrect data type. Are you using the 'dt.day_name()' function?"
    assert set(answer['weekday_aired']) == {'Monday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'}, "The 'weekday_aired' colum has incorrect values. Are you using the 'dt.day_name()' function?"
    return("Success")

def test_2f(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "215bb47da8fac3342b858ac3db09b033c6c46e0b", "Your answer is incorrect.\
    Are you summing up all rows with a 'weekday_aired ' column value of Tuesday?"
    return("Success")

def test_2g(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "77de68daecd823babbb58edb1c8e14d7106e83bb", "Your answer is incorrect. Are you grouping by a column? Are you using a 'for' loop?\
    The '.min() and .max() functions may be useful here."
    return("Success")

def _run_cleaner(answer, dirty):
    assert not answer is None, "Your function does not exist. Have you passed in the correct function?"
    try:
        res = answer(dirty.copy())
    except Exception as err:
        raise AssertionError(
            "Your function raised an error when we ran it on 'dirty': {0}".format(err))
    assert not res is None, "Your function does not return anything. Make sure it returns the cleaned dataframe."
    assert isinstance(res, pd.DataFrame), "Your function should return a dataframe."
    assert res.shape[0] == dirty.shape[0], "Your cleaned dataframe should still have {0} rows, but it has {1}.".format(
        dirty.shape[0], res.shape[0])
    return res


def _has_padding(series):
    return series.notnull() & (series.astype(str) != series.astype(str).str.strip())


def _check_fix(answer, dirty, clean, column, mask, what):
    res = _run_cleaner(answer, dirty)
    assert column in res.columns, "Your cleaned dataframe is missing the '{0}' column.".format(
        column)
    idx = list(dirty.index[mask])
    assert len(idx) > 0, "Grading error: no rows of 'dirty' show {0}. Please contact your instructor.".format(
        what)
    missing = [i for i in idx if i not in res.index]
    assert not missing, "Your cleaned dataframe no longer contains all of the original rows, so {0} cannot be checked.".format(
        what)
    wrong = [(i, res.at[i, column], clean.at[i, column]) for i in idx
             if res.at[i, column] != clean.at[i, column]]
    if wrong:
        i, got, want = wrong[0]
        extra = "" if len(wrong) == 1 else " ({0} other row(s) are also wrong.)".format(
            len(wrong) - 1)
        raise AssertionError(
            "You have not fixed {0}. In row {1} of '{2}' we expected {3!r} but got {4!r}.{5}".format(
                what, i, column, want, got, extra))
    return ("Success")


def test_3a(answer, dirty, clean):
    res = _run_cleaner(answer, dirty)
    assert list(res.columns) == list(clean.columns), "The columns of your cleaned dataframe are not in the same order as 'clean'. Expected {0} but got {1}.".format(
        list(clean.columns), list(res.columns))
    return ("Success")


def test_3b(answer, dirty, clean):
    return _check_fix(answer, dirty, clean, 'continent',
                      dirty['continent'].isnull(),
                      "the missing values in 'continent'")


def test_3c(answer, dirty, clean):
    return _check_fix(answer, dirty, clean, 'continent',
                      _has_padding(dirty['continent']),
                      "the extra whitespace around values in 'continent'")


def test_3d(answer, dirty, clean):
    return _check_fix(answer, dirty, clean, 'country',
                      _has_padding(dirty['country']),
                      "the extra whitespace around values in 'country'")


def test_3e(answer, dirty, clean):
    return _check_fix(answer, dirty, clean, 'country',
                      dirty['country'] == 'Central african republic',
                      "the capitalisation of 'Central african republic'")


def test_3f(answer, dirty, clean):
    return _check_fix(answer, dirty, clean, 'country',
                      dirty['country'] == 'china',
                      "the capitalisation of 'china'")


def test_3g(answer, dirty, clean):
    return _check_fix(answer, dirty, clean, 'country',
                      dirty['country'].isin(['Congo, Democratic Republic',
                                             'Democratic Republic of the Congo']),
                      "the two different spellings of 'Congo, Dem. Rep.'")


def test_3h(answer, dirty, clean):
    return _check_fix(answer, dirty, clean, 'country',
                      dirty['country'] == "Cote d'Ivore",
                      "the spelling of \"Cote d'Ivoire\"")


def test_2g(answer):
    assert not answer is None, "Your object does not exist. Have you passed in the correct object?"
    assert sha1(str(answer).encode('utf8')).hexdigest() == "77de68daecd823babbb58edb1c8e14d7106e83bb", "Your answer is incorrect. Please try again"
    return("Success")

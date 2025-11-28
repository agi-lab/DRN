import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from drn import *


# Sample test data and scenarios we discussed
@pytest.mark.parametrize(
    "L_Raw, data_train, M, expected",
    [
        # Test Case 1: Small dataset with clear cutpoints
        (
            [0, 3, 6, 9],
            np.array([0.5, 2.5, 4.5, 6.5, 8.5]),
            2,
            [0, 3, 9],
        ),  # As derived from our correct implementation
        # Test Case 2: Problematic case for the old algorithm where the final bucket might have less than M observations
        (
            [0, 3, 7, 9],
            np.array([0.5, 2.5, 4.5, 6.5, 8.5]),
            2,
            [0, 3, 9],
        ),  # Adjusted expectation to fit the correct algorithm output
    ],
)
def test_merge_cutpoints(L_Raw, data_train, M, expected):
    assert merge_cutpoints(L_Raw, data_train, M) == expected


def test_split_preprocess():
    # Create synthetic dataset
    rng = np.random.RandomState(42)
    n_samples = 100

    # Numerical features
    num1 = rng.normal(loc=0, scale=1, size=n_samples)
    num2 = rng.uniform(low=0, high=10, size=n_samples)

    # Categorical features
    cat_choices = ["A", "B", "C"]
    cat1 = rng.choice(cat_choices, size=n_samples)
    cat2 = rng.choice(["X", "Y"], size=n_samples)

    # Combine into DataFrame
    features = pd.DataFrame({"num1": num1, "num2": num2, "cat1": cat1, "cat2": cat2})

    # Create a target variable correlated with num1
    target = pd.Series(num1 * 2 + rng.normal(scale=0.5, size=n_samples), name="target")

    # Define feature lists
    cat_features = ["cat1", "cat2"]
    num_features = ["num1", "num2"]

    # 3) Combined split_and_preprocess
    (
        x_train1,
        x_val1,
        x_test1,
        y_train1,
        y_val1,
        y_test1,
        x_train_raw1,
        x_val_raw1,
        x_test_raw1,
        num_features1,
        cat_features1,
        all_categories1,
        ct1,
    ) = split_and_preprocess(
        features,
        target,
        num_features=num_features,
        cat_features=cat_features,
        num_standard=True,
        seed=42,
    )

    # 4) Separate split then preprocess
    x_train_raw2, x_val_raw2, x_test_raw2, y_train2, y_val2, y_test2 = split_data(
        features, target, seed=42
    )
    x_train2, x_val2, x_test2, ct2, all_categories2 = preprocess_data(
        x_train_raw2,
        x_val_raw2,
        x_test_raw2,
        num_features=num_features,
        cat_features=cat_features,
        num_standard=True,
    )

    assert (x_train1.index == x_train2.index).all()
    assert (x_val1.index == x_val2.index).all()
    assert (x_test1.index == x_test2.index).all()
    assert (y_train1.index == y_train2.index).all()
    assert (y_val1.index == y_val2.index).all()
    assert (y_test1.index == y_test2.index).all()

    # 5) Assert raw splits are identical
    pdt.assert_frame_equal(x_train_raw1, x_train_raw2)
    pdt.assert_frame_equal(x_val_raw1, x_val_raw2)
    pdt.assert_frame_equal(x_test_raw1, x_test_raw2)
    pdt.assert_series_equal(y_train1, y_train2)
    pdt.assert_series_equal(y_val1, y_val2)
    pdt.assert_series_equal(y_test1, y_test2)

    # 6) Assert processed features are numerically equal
    np.testing.assert_allclose(x_train1, x_train2, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(x_val1, x_val2, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(x_test1, x_test2, rtol=1e-6, atol=1e-8)

    # 7) Assert categories metadata match
    for feature in all_categories1:
        assert sorted(list(all_categories1[feature])) == all_categories2[feature]

    assert len(all_categories1) == len(all_categories2)


def test_basic_replacement():
    df = pd.DataFrame(
        {
            "animal": ["cat", "dog", "dog", "lion", "lion", "lion", "tiger"],
            "count": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    result = replace_rare_categories(df, threshold=3)
    assert result["animal"].tolist() == [
        "OTHER",
        "OTHER",
        "OTHER",
        "lion",
        "lion",
        "lion",
        "OTHER",
    ]
    assert isinstance(result["animal"].dtype, pd.CategoricalDtype)


def test_no_replacement_needed():
    df = pd.DataFrame({"color": ["red", "red", "blue", "blue", "green", "green"]})
    result = replace_rare_categories(df, threshold=2)
    assert set(result["color"]) == {"red", "blue", "green"}


def test_all_rare_categories():
    df = pd.DataFrame({"tool": ["hammer", "screwdriver", "wrench"]})
    result = replace_rare_categories(df, threshold=2)
    assert set(result["tool"]) == {"OTHER"}
    assert all(val == "OTHER" for val in result["tool"])
    assert isinstance(result["tool"].dtype, pd.CategoricalDtype)


def test_custom_column_selection():
    df = pd.DataFrame(
        {
            "cat_col": ["a", "b", "b", "c", "c", "c"],
            "other_col": ["x", "x", "y", "y", "z", "z"],
        }
    )
    result = replace_rare_categories(df, threshold=3, cat_features=["cat_col"])
    assert set(result["cat_col"]) == {"OTHER", "c"}
    assert set(result["other_col"]) == {"x", "y", "z"}  # untouched


def test_non_categorical_data_ignored():
    df = pd.DataFrame(
        {"cat": ["a", "b", "b", "c", "c", "c"], "num": [1, 2, 3, 4, 5, 6]}
    )
    result = replace_rare_categories(df, threshold=3)
    assert result["num"].equals(df["num"])  # unchanged
    assert isinstance(result["cat"].dtype, pd.CategoricalDtype)


def test_placeholder_conflict_raises_error():
    df = pd.DataFrame({"animal": ["cat", "dog", "OTHER", "lion", "lion", "lion"]})
    with pytest.raises(ValueError, match="placeholder value 'OTHER' already exists"):
        replace_rare_categories(df, threshold=2, placeholder="OTHER")

    # Another way this may occur is if the function is called twice by accident
    df = pd.DataFrame(
        {
            "animal": ["cat", "dog", "dog", "lion", "lion", "lion", "tiger"],
            "count": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    result = replace_rare_categories(df, threshold=3)
    with pytest.raises(ValueError, match="placeholder value 'OTHER' already exists"):
        replace_rare_categories(result, threshold=2)

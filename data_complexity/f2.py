from typing import Union, Literal
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def overlapping_area(c1: pd.DataFrame, c2: pd.DataFrame) -> float:
    _min = np.vstack((c1.min(), c2.min()))
    _max = np.vstack((c1.max(), c2.max()))

    nominator = np.clip(np.min(_max, axis=-2) - np.max(_min, axis=-2), a_min=0.0, a_max=np.inf)
    denominator = np.max(_max, axis=-2) - np.min(_min, axis=-2)

    return float(np.prod(np.divide(nominator, denominator)))


def f2(
        features: pd.DataFrame,
        target: pd.DataFrame,
        agg: Union[Literal["sum"], Literal["mean"], Literal["max"]] = "mean",
        *args, **kwargs
) -> float:
    """
    calculates area of overlapping regions (F2 measure) for the provided data. Calculations are based on:
        https://arxiv.org/pdf/1808.03591.pdf

    :param features: a dataframe containing numerical features
    :param target: a dataframe containing target classes
    :param agg: a function to aggregate partial areas of overlapping regions
    :return: F2 score
    """

    assert features.shape[0] == target.shape[0], 'Features and target are required to have the same number of instances'

    assert len(features.columns) == len(
        features.select_dtypes([np.number]).columns), 'Only numerical features are accepted'
    target_name = target.columns[0]

    class_names = target[target_name].unique()
    assert len(class_names) >= 2, "Can't compute F2 for one class"

    partials = list()
    for c1, c2 in combinations(class_names, 2):
        _c1, _c2 = features[target["target"] == c1], features[target["target"] == c2]
        partials.append(overlapping_area(_c1, _c2))
    _f2: float = getattr(np, agg)(np.array(partials))
    return _f2


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True, as_frame=True)
    y = y.to_frame()

    score = f2(X, y, agg="mean")
    print(f'F2 measure: {score}')

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


def f1(features: pd.DataFrame, target: pd.DataFrame) -> float:
    """
    calculates Maximum Fisher's Discriminant Ratio (F1 measure) for the provided data.
    calculations are based on:
        https://arxiv.org/pdf/1808.03591.pdf

    :param features: a dataframe containing numerical features describing the data
    :param target: a dataframe containing target classes of the data
    :return: computed F1 measure
    """
    assert features.shape[0] == target.shape[0], 'Features and target are required to have the same number of instances'

    assert len(features.columns) == len(features.select_dtypes([np.number]).columns), 'Only numerical features are accepted'
    target_name = target.columns[0]
    n_cj = target.groupby(target_name)[target_name].count()

    assert len(list(n_cj.index)) >= 2, "Can't compute F1 for one class"

    df = features.join(target)
    mean_c = df.groupby(target_name).mean()
    mean_f = features.mean()

    nominator = n_cj.T * ((mean_c - mean_f) ** 2).T
    nominator = nominator.T.sum()

    mean_c[target_name] = 0
    denominator = df.apply(lambda x: (x - mean_c.loc[x[target_name]]) ** 2, axis=1)
    denominator = denominator.drop(columns=[target_name]).sum()

    r = nominator / denominator
    r[~np.isfinite(r)] = 0

    assert np.all(r > 0), 'All the instances are equal except for the class.'

    return 1 / (1 + np.max(r))


if __name__ == '__main__':
    X, y = load_iris(as_frame=True, return_X_y=True)
    y = y.to_frame()
    f1 = f1(X, y)

    print(f'F1 measure: {f1}')

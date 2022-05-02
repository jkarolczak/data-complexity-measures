import collections

import pandas as pd
import numpy as np


def fisher_discriminant_ratio(features: pd.DataFrame, target: pd.DataFrame, ) -> float:
    """

    :param features:
    :param target:
    :return:
    """
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



# for tests
if __name__ == '__main__':
    pass

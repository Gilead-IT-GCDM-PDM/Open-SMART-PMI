import pandas as pd
import pickle
import os

import train


PERSISTED_MODEL = '_persisted_model.pkl'


def rf_using_old_attrs():
    data = pd.read_csv('../data/ci5001778_si_001.txt')
    if os.path.exists(PERSISTED_MODEL):
        with open(PERSISTED_MODEL, 'rb') as f:
            m = pickle.load(f)
    else:
        m = train.rf_regress(data.iloc[:, 6:], data.loc[:, 'meanComplexity'], grid_search=True)
        with open(PERSISTED_MODEL, 'wb') as f:
            pickle.dump(m, f)

    # dict_m = {'rf': m}
    # train.compute_model_scores(data.iloc[:, 6:], data.loc[:, 'meanComplexity'], dict_m)
    # train.make_predictions(m, data.iloc[:301, 6:], data.loc[:300, 'meanComplexity'])
    train.make_predictions(m, data.iloc[:, 6:], data.loc[:, 'meanComplexity'])


if __name__ == '__main__':
    rf_using_old_attrs()


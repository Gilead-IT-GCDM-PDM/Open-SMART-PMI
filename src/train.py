import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler


import molecular_descriptors


# -- VARIABLE SETTING
basemodels = {
    "tuned_rf": RandomForestRegressor(
        max_depth=6,
        max_features=None,
        min_samples_leaf=2,
        min_samples_split=30,
        n_estimators=366,
        random_state=42
    ),
    "rf": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "nnet": MLPRegressor(  # must be normalized
        random_state=42,
        max_iter=500
    ),
    "svr": SVR(
        gamma='auto'
    ),
    "pls": PLSRegression(  # must be normalized
        n_components=3,
        scale=True
    ),
    "knn": KNeighborsRegressor(
        n_neighbors=7,
        weights='distance'
    ),
}

s = 300


def train(data_path, output_dir, grid_search=True):
    data = read_data(data_path)
    # data = data[:20]

    assert 'SMILES' in data.columns, "Missing SMILES column in data"
    assert 'meanComplexity' in data.columns, "Missing meanComplexity column in data"

    # X, y = prepare_data_for_training(data)
    if os.path.exists('X.pkl'):
        X = pickle.load(open('X.pkl', 'rb'))
    else:
        X = molecular_descriptors.compute(data.SMILES)
    y = data.meanComplexity.loc[X.index]

    X.fillna(0, inplace=True)

    missing_cols = filter_columns(X)
    X.drop(columns=missing_cols, inplace=True)
    # should happen before prepare
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    low_var_cols = filter_columns_training_data(X_train)
    X_train.drop(columns=low_var_cols, inplace=True)
    X_test.drop(columns=low_var_cols, inplace=True)

    model = rf_regress(X_train, y_train, grid_search)
    # model = result['model']
    make_predictions(model, X_test, y_test)
    # print evaluation using X_test, t_test
    # test model
    now = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    pickle.dump(model, open(f'{output_dir}/model_{now}', 'wb'))
    pickle.dump(low_var_cols.union(missing_cols), open(f'{output_dir}/missing_cols', 'wb'))


def read_data(filename):
    data = pd.read_csv(filename)
    return data

"""

def prepare_data_for_training(data):
    assert 'SMILES' in data.columns, "Expected SMILES columns in data"
    assert 'meanComplexity' in data.columns, "Expected meanComplexity in data"

    X = molecular_descriptors.compute(data.SMILES)
    columns_to_drop = filter_columns(X)
    X = X.drop(columns=['SMILES'] + columns_to_drop).fillna(0),

    low_variance = filter_training_set_columns(X)
    X.drop(columns=low_variance, inplace=True)
    y = data.meanComplexity.loc[X.index]

    return X, y
"""


"""
def old_prepare(data):
    X = desc.molecular_descriptors(data.SMILES)

    columns_to_drop = filter_columns(X)

    y = data.meanComplexity.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=['SMILES']+columns_to_drop).fillna(0),
        y,
        test_size=0.2,
        random_state=42
    )

    low_variance = filter_training_set_columns(X_train)
    X_train = X_train.drop(columns=low_variance)
    # X_test = X_test.drop(columns=low_variance)
    return X, y, X_train, X_test, y_train, y_test
"""


def filter_columns(df):
    """ dispose of columns that are null or error objects
    """
    nonnumerics = df.copy().drop(columns=['SMILES']).select_dtypes(exclude='number').applymap(type).apply(set).to_frame()
    ind = pd.get_dummies(nonnumerics.explode(0))
    errors_mask = [int('Error' in c) for c in ind.columns]
    missing_mask = [int('Missing' in c) for c in ind.columns]
    errors = ind[ind == errors_mask].dropna().index
    missings = ind[ind == missing_mask].dropna().index

    nulls = set(errors).union(set(missings)).union({'SMILES'})

    return nulls


def filter_columns_training_data(df):
    dff = df.astype(float)
    m = len(dff.columns)
    null_rate = 0.125
    gt125pct_null = dff.columns[(dff.isna().sum() / m) > null_rate]
    gt125pct_null = set(gt125pct_null)

    scaler = MinMaxScaler()
    xp = scaler.fit_transform(dff)
    xp = pd.DataFrame(xp, columns=dff.columns)
    low_var = xp.var(axis=0) <= 0.00001
    low_var_columns = xp.columns[low_var == True]
    low_var_columns = set(low_var_columns)

    cols_to_drop = low_var_columns.union(gt125pct_null)
    return cols_to_drop


def compute_model_scores(x_train, y_train, models: dict):
    scores = {}
    model_info = {}

    for m in models:
        scores[m] = {}

        # train model and predict on val set
        models[m].fit(x_train, y_train)
        y_pred = models[m].predict(x_train)

        # calculate cross_validated error metrics
        # more metrics in PMI notebooks

        cv_score = cross_val_score(models[m], x_train, y_train, scoring='r2', cv=5)
        scores[m]['CV-R2'] = np.mean(cv_score)

        rmse_score = cross_val_score(models[m], x_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
        scores[m]['CV-RMSE'] = np.mean(rmse_score)

        scores[m] = {k: np.around(v, 3) for k, v in scores[m].items()}

        model_info[m] = models[m]

    scores_df = pd.DataFrame(scores).T
    return scores_df, model_info


def rf_regress(train_x, train_y, val_split=0.8, grid_search=False):
    # -- Pick model
    model = base_model()

    if grid_search:
        # -- Parameter Tuning
        random_grid = {}
        # Maximum number of levels in tree
        max_depth = range(1, 7)
        random_grid['max_depth'] = max_depth
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [2, 4, 6, 10]
        random_grid['min_samples_leaf'] = min_samples_leaf
        # Number of trees, reduce overfitting
        n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
        random_grid['n_estimators'] = n_estimators

        # # Number of trees in random forest
        # n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
        # random_grid['n_estimators'] = n_estimators
        # Number of features to consider at every split
        max_features = [None, 'sqrt', 'log2']
        random_grid['max_features'] = max_features
        # Minimum number of samples required to split a node
        min_samples_split = [5, 10, 20, 30]
        random_grid['min_samples_split'] = min_samples_split
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        random_grid['bootstrap'] = bootstrap

        # FIXME: tune the scoring method: https://gabrieltseng.github.io/posts/2018-02-25-XGB/
        model = RandomizedSearchCV(
            scoring='neg_root_mean_squared_error',
            estimator=model,
            param_distributions=random_grid,
            n_iter=40,
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        # x = x.apply(lambda i: np.log10(i) if np.issubdtype(type(i), np.number) else i)
    train_x = train_x.astype(float)
    model.fit(train_x, train_y)

    now = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    pickle.dump(model, open(f'../data/model_{now}', 'wb'))


    return model


def make_predictions(model, X, y):
    # print('model parameters', standard_model.get_params())
    print('...Validation...')
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    rmse = mse ** .5
    r2 = r2_score(y, pred)
    scores = cross_val_score(model, X, y, scoring='r2', cv=5)
    rmse_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)

    print(' MAE', mae)
    print(' MSE', mse)
    print(' RMSE', rmse)
    print(' R^2', r2)
    print(' CV-R^2', scores)
    print(' avg CV-R^2', np.mean(scores))
    print(' avg CV-RMSE', np.mean(rmse_scores))


def base_model():
    """ With the optimal hyperparameters for the training data
    """
    return RandomForestRegressor(
        max_depth=6,
        max_features='log2',
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=472,
        random_state=42
    )



# -- STANDARD IMPORTS --
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# -- PACKAGE IMPORTS --
import molecular_descriptors
from utilities import read_file


# -- TRAINING MODULE --

def train(data_path: str, output_dir: str, grid_search=True) -> None:
    '''
    Main training loop to preprocess the data and train a model

    Arguments
    ---
    `data_path`: path to compound file SDFs or SMILES
    `output_dir`: path to store model and experiment results
    `grid_search`: perform grid search for hyperparameter tuning
    '''
    # create output directory
    data = read_file(data_path)
    now = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    path = os.path.join(output_dir, 'model_'+now)
    os.mkdir(path=path)

    X = molecular_descriptors.compute(data.SMILES)
    y = data.meanComplexity.loc[X.index]

    # data preprocessing
    X.fillna(0, inplace=True)
    removed_cols = filter_columns(X)
    X.drop(columns=removed_cols, inplace=True)
    # should happen before prepare
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    low_var_cols = filter_test_rm_low_var(X_train)
    X_train.drop(columns=low_var_cols, inplace=True)
    X_test.drop(columns=low_var_cols, inplace=True)

    # make predictions
    model = rf_regress(X_train, y_train, grid_search)
    make_predictions(model, X_test, y_test)

    # write results and variables to folder
    model_aattrs = [model, X_test.columns]
    removed = low_var_cols.union(removed_cols)
    write_data(path, now,
               model_aattrs,
               X_train, y_train, X_test, y_test,
               removed)


# -- DATA FILTERING -- 

def filter_columns(df) -> set[str]:
    """ 
    Return set of columns that contain largely null or error objects
    """
    nonnumerics = df.copy().drop(columns=['SMILES']).select_dtypes(exclude='number').applymap(type).apply(set).to_frame()
    ind = pd.get_dummies(nonnumerics.explode(0))
    errors_mask = [int('Error' in c) for c in ind.columns]
    missing_mask = [int('Missing' in c) for c in ind.columns]
    errors = ind[ind == errors_mask].dropna().index
    missings = ind[ind == missing_mask].dropna().index

    nulls = set(errors).union(set(missings)).union({'SMILES'})

    return nulls


def filter_test_rm_low_var(df) -> set[str]:
    '''
    Use training set data to remove variables of low variance
    '''
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

# -- EXPERIMENT TRACKING -- 

def write_data(path, now,
               model_aattrs,
               X_train, y_train, X_test, y_test,
               removed):
    '''
    Save variables to experiment folder
    '''
    pickle.dump(model_aattrs, open(f'{path}/{now}_model_aattrs.pkl', 'wb'))
    pickle.dump(X_train, open(f'{path}/x_train.pkl', 'wb'))
    pickle.dump(y_train, open(f'{path}/y_train.pkl', 'wb'))
    pickle.dump(X_test, open(f'{path}/x_test.pkl', 'wb'))
    pickle.dump(y_test, open(f'{path}/y_test.pkl', 'wb'))
    pickle.dump(removed, open(f'{path}/removed_cols', 'wb'))
    print(f'... Saved to <{path}>')



# -- GENERAL MODELING HELPER FUNCTIONS --
    
def make_predictions(model, X, y):
    '''
    Compute and print model metrics to terminal
    '''
    print('... Model Metrics ...')
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


def compute_model_scores(x_train, y_train, models: dict):
    '''
    Generalized helper function to quickly evaluate models
    '''
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
    '''
    Helper function to quickly tune and evaluate random forest models
    '''

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

    train_x = train_x.astype(float)
    model.fit(train_x, train_y)

    return model


def base_model():
    """ 
    With the optimal hyperparameters for the full training data
    """
    return RandomForestRegressor(
        max_depth=6,
        max_features='log2',
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=472,
        random_state=42
    )



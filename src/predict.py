import os
import pickle
import traceback
import pandas as pd
import openpyxl
from glob import glob
from datetime import datetime
from rdkit.Chem import PandasTools


import molecular_descriptors


"""
def make_predictions(filenames):
    # -- processing new molecules

    df = read_sdf_files(filenames)
    X_new = prepare_data(df.SMILES)

    gs_04_model = pickled_model()

    df['Predictions'] = gs_04_model.predict(X_new[['UNIQUETT', 'NumAtomStereoCenters', 'NumHeteroatoms', 'chi4n',]])

    # FIXME: add mol wt and smart-pmi conversion
    # output rpredictions
    return df
"""


def read_sdf_files(names):
    dfs = []
    # convert SDF to SMILES
    for file in names:
        load_mol = PandasTools.LoadSDF(file, smilesName='SMILES').head(1)
        dfs += [load_mol]

    df = pd.concat(dfs)
    df['ID'] = names
    df.reset_index(drop=True, inplace=True)
    return df


"""
def prepare_data(data):
    assert set(['SMILES', 'meanComplexity']) <= set(data.columns)
    y_new = data.meanComplexity.loc[X_new.index]

    return X_new, y_new
"""

def predict(model_path, input, output_dir):
    '''
    predict the complexity and SMART-PMI of a given molecule(s). 

    `model_path`: path to the chosen model object
    `input`: the molecule source. `input` is compatible with:
        (1) a directory of .SDF files 
        (2) a file containing SMILES strings (denoted by 'SMILES' in the header row)
        (3) a pandas dataframe containing SMILES strings as a pickled object
    `output`: directory to store prediction information

    '''
    assert os.path.exists(model_path), f"Model: {model_path} does not exists"
    assert os.path.exists(output_dir), f"Output dir: {output_dir} does not exists"

    try:
        input_ext = input[-4:]
        # path for predicting from directory of SDF files 
        if os.path.isdir(input):
            assert os.path.exists(input_dir), f"Input dir: {input_dir} does not exists"
            input_dir = input
            filenames = list(glob(f"{input_dir}/*.sdf"))
            df = read_sdf_files(filenames)
        # path for predicting from csv or excel file
        elif ('.csv' in input_ext) | ('.txt' in input_ext):
            df = pd.read_csv(input)
        elif ('.xlsx' in input_ext):
            df = pd.read_excel(input)
        # path for predicting from pickled dataframe
        elif ('.obj' in input_ext) | ('.pkl' in input_ext):
            with open(input, 'rb') as f:
                df = pickle.load(f)

        assert 'SMILES' in df.columns, "Missing SMILES column in data"
            
        predicts = make_predictions(model_path, df)
        generate_output(predicts, output_dir)
        return predicts
    except:
        traceback.print_exc()


def make_predictions(model_path, df):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X = molecular_descriptors.compute(df.SMILES)

    model_params = list(pd.read_csv('../model/model_params.csv').params)
    df['Predictions'] = model.predict(X[model_params].astype(float))

    # mol wt = (processed dataset instance).amw
    # smart_pmi = (0.13 * .amw) + (177 * [complexity]) - 252


    # TODO add molecular weight etc
    return df


def generate_output(df, output_dir):
    now = datetime.now().strftime('%y-%m-%d-%H%M%S')
    output_path = os.path.join(output_dir, 'predictions_{now}.csv')
    df.to_csv(output_path, index=False)  #, engine=openpyxl)

# think about docker for webapp

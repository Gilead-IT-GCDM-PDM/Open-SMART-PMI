# -- STANDARD IMPORTS --
import os
import pickle
import traceback
import pandas as pd
from glob import glob
from datetime import datetime
import rdkit.Chem
from rdkit.Chem import Draw, PandasTools

# -- PACKAGE IMPORTS -- 
import molecular_descriptors


# -- Set path for Model GS-04

MODEL_GS_04 = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    'models',
    'GS_04_Model_attrs.obj'
)

# -- PREDICTION MODULE

def calculate_smart_pmi(r):
    '''
    Calculation of SMART-PMI score using complexity
    '''
    mol_wt = r['MW']
    complexity = r['Complexity']
    smart_pmi = (0.13 * mol_wt) + (177 * complexity) - 252
    return round(smart_pmi, 2)


def predict(input, output_dir, model_path=''):
    '''
    predict the complexity and SMART-PMI of a given molecule(s). 

    `model_path`: path to the chosen model object
    `input`: the molecule source with. `input` is compatible with:
        (1) a directory of .SDF files 
        (2) a csv/excel file containing SMILES strings (denoted by 'SMILES' in the header row)
        (3) a pickled object containing pandas dataframe containing SMILES strings (denoted by 'SMILES' in the header row)
    `output`: directory to store prediction information

    '''
    assert os.path.exists(output_dir), f"Output dir: {output_dir} does not exists"

    try:
        df = read_input(input=input)

        assert 'SMILES' in df.columns, "Missing SMILES column in data"

        if not model_path:
            model_path = MODEL_GS_04
        
        preds = make_predictions(df, model_path)
        generate_output(preds, output_dir)
        return preds
    except:
        traceback.print_exc()


def make_predictions(x, model_path=MODEL_GS_04):
    '''
    Use given model to make molecular complexity and SMART-PMI predictions
    '''
    with open(model_path, 'rb') as f:
        model, attributes = pickle.load(f)

    # FIXME: data needs to go through SAME preprocessing steps as training
    smiles = x.SMILES
    X = molecular_descriptors.compute(smiles)

    res = pd.DataFrame()
    res['molwt'] = X.exactmw
    res['molComplexity'] = model.predict(X[attributes].astype(float))
    res['SMART-PMI'] = (0.13 * res.molwt) + (177 * res.molComplexity) - 252
    res['SMILES'] = smiles

    return res.round(3)


def read_sdf_files(names):
    dfs = []
    # convert SDF to SMILES
    for file in names:
        load_mol = rdkit.Chem.PandasTools.LoadSDF(file, smilesName='SMILES').head(1)
        dfs += [load_mol]

    df = pd.concat(dfs)
    df['ID'] = names
    df.reset_index(drop=True, inplace=True)
    
    return df

def read_input(input):
        input_ext = input[-4:]

        # path for predicting from directory of SDF files 
        if os.path.isdir(input):
            assert os.path.exists(input), f"Input dir: {input} does not exists"
            input_dir = input
            filenames = list(glob(f"{input_dir}/*.sdf"))
            df = read_sdf_files(filenames)

        # path for predicting from csv or excel file
        elif ('.csv' in input_ext) | ('.txt' in input_ext):
            df = pd.read_csv(input)
        elif ('.xls' in input_ext):
            df = pd.read_excel(input)

        # path for predicting from pickled dataframe
        elif ('.obj' in input_ext) | ('.pkl' in input_ext):
            with open(input, 'rb') as f:
                df = pickle.load(f)

        return df


def generate_output(df, output_dir):
    now = datetime.now().strftime('%y-%m-%d-%H%M%S')
    output_path = os.path.join(output_dir, f'predictions_{now}.csv')
    df.to_csv(output_path, index=False)  #, engine=openpyxl)
    print(f'... Writing predictions to {output_path}')

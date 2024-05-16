# -- STANDARD IMPORTS --
import os
import pickle
import traceback
import pandas as pd
from glob import glob
from datetime import datetime
import rdkit.Chem

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

def calculate_smart_pmi(r:pd.DataFrame) -> pd.DataFrame:
    '''
    Calculation of SMART-PMI score using complexity
    '''
    mol_wt = r['MW']
    complexity = r['Complexity']
    smart_pmi = (0.13 * mol_wt) + (177 * complexity) - 252
    return round(smart_pmi, 2)


def predict(filepath: str, output_dir: str, model_path: str='') -> pd.DataFrame:
    '''
    predict the complexity and SMART-PMI of a given molecule(s). 

    Arguments:
    ---
    `model_path`: path to the chosen model object
    `filepath`: the molecule source with. `filepath` is compatible with:
        (1) a directory of .SDF files 
        (2) a csv/excel file containing SMILES strings (denoted by 'SMILES' in the header row)
        (3) a pickled object containing pandas dataframe containing SMILES strings (denoted by 'SMILES' in the header row)
    `output`: directory to store prediction information

    '''
    assert os.path.exists(output_dir), f"Output dir: {output_dir} does not exists"

    try:
        df = read_file(filepath=filepath)

        assert 'SMILES' in df.columns, "Missing SMILES column in data"

        if not model_path:
            model_path = MODEL_GS_04
        
        preds = make_predictions(df, model_path)
        generate_output(preds, output_dir)
        return preds
    except:
        traceback.print_exc()


def make_predictions(x, model_path=MODEL_GS_04) -> pd.DataFrame:
    '''
    Use given model to make molecular complexity and SMART-PMI predictions
    '''
    with open(model_path, 'rb') as f:
        model, attributes = pickle.load(f)

    # FIXME: data needs to go through SAME preprocessing steps as training
    smiles = x.SMILES
    X = molecular_descriptors.compute(smiles)

    res = pd.DataFrame()
    res['MW'] = X.exactmw
    res['COMPLEXITY'] = model.predict(X[attributes].astype(float))
    res['SMART-PMI'] = (0.13 * res.MW) + (177 * res.COMPLEXITY) - 252
    res['SMILES'] = smiles
    res['FILENAME'] = x['FILENAME']
    res['NAME'] = x['NAME']
    res['ROMol'] = x['ROMol'].apply(lambda x: extract_img_src(x))
    return res.round(3)


def extract_img_src(x):
    # img_tag will be of format <img data-content="..." src="..." alt="..."/>
    img_tag = f'{x}'
    start = img_tag.find("src=")
    end = img_tag.find(" alt")
    src = img_tag[start+4: end]
    return src


def generate_output(df, output_dir) -> None:
    now = datetime.now().strftime('%y-%m-%d-%H%M%S')
    output_path = os.path.join(output_dir, f'predictions_{now}.csv')
    df.to_csv(output_path, index=False)
    print(f'... Writing predictions to {output_path}')

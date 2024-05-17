import os
import time
import base64
import pandas as pd
from glob import glob
from rdkit.Chem import PandasTools


def persist_sdf_data(content):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    with open('tmp/uploaded.sdf', 'wb') as f:
        f.write(decoded)
    return decoded


def read_sdf_data(contents, names):
    filenames = []
    for content, name in zip(contents, names):
        print(f'processing: {name}')
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        os.makedirs('tmp', exist_ok=True)
        random_dir = int(time.monotonic()*10000)
        os.makedirs(f'tmp/{random_dir}')
        filename = os.path.join(f'tmp/{random_dir}', name)
        with open(filename, 'wb') as f:
            f.write(decoded)
        filenames += [filename]

    df = read_sdf_files(filenames)
    return df


def read_sdf_files(names):
    dfs = []
    # convert SDF to SMILES
    for file in names:
        load_mol = PandasTools.LoadSDF(file, smilesName='SMILES')  # .head(1)
        load_mol['FILENAME'] = os.path.basename(file)
        if 'NAME' not in load_mol.columns:
            load_mol['NAME'] = load_mol['FILENAME']
        dfs += [load_mol]

    df = pd.concat(dfs, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    return df


def read_file(filepath: str) -> pd.DataFrame:
    assert os.path.exists(filepath), f"Input dir: {filepath} does not exists"
    ext = filepath[-4:]
    # read directory of SDF files
    if os.path.isdir(filepath):
        filenames = list(glob(f"{filepath}/*.sdf"))
        df = read_sdf_files(filenames)
    # read csv or excel file
    elif '.csv' in ext or '.txt' in ext:
        df = pd.read_csv(filepath)
    elif '.xlsx' in ext:
        df = pd.read_excel(filepath)

    return df

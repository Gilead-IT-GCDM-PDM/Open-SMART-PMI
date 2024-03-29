import numpy as np
import pandas as pd
import traceback

from joblib import dump, load
import dill as pickle
import sys, os
import random
import re

# from rdkit import Chem
# from rdkit.Chem import PandasTools
# from mordred import Calculator, descriptors

from models.molecular_featurizer import *


model = load('models/final_RFmodel.joblib')
features = load('models/final_RFfeatures.joblib')

filenames = 'testdata.sdf'
mandeep_obj = 'output.csv'


def predict_structure(filename, mandeep_obj):
    gs_smis = PandasTools.LoadSDF(filename, smilesName='SMILES').head(1)
    gs_smis['ID'] = filename
    gs_smis = gs_smis.reset_index(drop=True)
    
    new = featurize(gs_smis, True, True).drop(['ID','SMILES','ROMol'], 1)
    
    # mandeep = pd.read_csv(mandeep_obj).drop('MOLECULE', 1).head(1)  #FIXME: MOE Attr file format
    mandeep = mandeep_obj.drop('MOLECULE', axis=1).head(1)
    mandeep['APCOMPLEX'] = mandeep['PCOMPLEX']
    mandeep['CAR_ALLATOM_RATIO'] = mandeep['AR_ALLATOM_RATIO']

    gs_test = pd.concat([mandeep, new], 1)[features]

    gs_smis['Predictions'] = np.round(model['Model'].predict(gs_test), 1)
    # gs_smis = gs_smis.drop('ROMol',1)
    gs_smis['MW'] = new.amw
    gs_smis['sPMI'] = (0.13 * gs_smis.MW) + (177 * gs_smis.Predictions) - 252
    return gs_smis


if __name__ == '__main__':
    print(predict_structure(filenames, mandeep_obj))
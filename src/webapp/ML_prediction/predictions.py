
import numpy as np
import pandas as pd
from joblib import dump, load
import dill as pickle
import sys, os
import random
import re

from rdkit import Chem
from rdkit.Chem import PandasTools
from mordred import Calculator, descriptors

from molecular_featurizer import *


model = load('smart_pmi/final_RFmodel.joblib')
features = load('smart_pmi/final_RFfeatures.joblib')

filenames = '/Users/nicolelrtin/Documents/smart_pmi/data/Compound files/AMB.sdf'
mandeep_obj = '/Users/nicolelrtin/Documents/smart_pmi/data/parsed_attributes_0907.csv'

def predict_structure(filename, mandeep_obj):
    gs_smis = PandasTools.LoadSDF(filename, smilesName='SMILES').head(1)
    
    gs_smis['ID'] = filename
    gs_smis = gs_smis.reset_index(drop=True)
    
    new = featurize(gs_smis, True, True).drop(['ID','SMILES','ROMol'],1)
    
    mandeep = pd.read_csv(mandeep_obj, header=1).drop('MOLECULE',1).head(1) #FIXME: MOE Attr file format
    gs_test = pd.concat([mandeep,new], 1)[features]
    
    gs_smis['Predictions'] = np.round(model['Model'].predict(gs_test),1)

    # gs_smis = gs_smis.drop('ROMol',1)
    gs_smis['MW'] = new.amw
    gs_smis['sPMI'] = (0.13 * gs_smis.MW) + (177 * gs_smis.Predictions) - 252
    return gs_smis

if __name__ == '__main__':
    print(predict_structure(filenames, mandeep_obj))
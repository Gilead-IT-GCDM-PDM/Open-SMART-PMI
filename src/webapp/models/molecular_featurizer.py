import numpy as np
import pandas as pd
import pickle
import sys, os
import random
import re

from rdkit import Chem
from rdkit.Chem import PandasTools
from mordred import Calculator, descriptors

# -- helper functions
np.random.seed(42)
random.seed(42)

# -- get RDKit featurizers
descriptor_names = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())
get_descriptors = Chem.rdMolDescriptors.Properties(descriptor_names)
# -- get Mordred featurizers
calc = Calculator(descriptors, ignore_3D=True)

def smi_to_descriptors(smile):
    # -- RDK helper func
    mol = Chem.MolFromSmiles(smile)
    descriptors = []
    if mol:
        # -- use RDKit featurizers
        descriptors = np.array(get_descriptors.ComputeProperties(mol))
    return descriptors 

# -- apply molecular featurization on SMILES
def featurize(df, use_rdk = False, use_m = False):
    df = df.copy()
    smiles = df.SMILES
    append = [df]

    # -- use RDKit featurizers
    if use_rdk:
        rdkit_desc = list(df.SMILES.apply(smi_to_descriptors))
        desc_df = pd.DataFrame(rdkit_desc, columns=descriptor_names)
        append += [desc_df]
    # -- use Mordred featurizers
    if use_m:
        mols = [Chem.MolFromSmiles(smi) for smi in smiles if Chem.MolFromSmiles(smi) != None]
        # drop = ['SpAbs_Dt',	'SpMax_Dt'	,'SpDiam_Dt'	,'SpAD_Dt',	'SpMAD_Dt'	,'LogEE_Dt'	,'SM1_Dt'	,'VE1_Dt',	'VE2_Dt',
	    #         'VE3_Dt',	'VR1_Dt',	'VR2_Dt',	'VR3_Dt'	,'DetourIndex']
        mord_df = calc.pandas(mols) #.select_dtypes(include=['int64', 'float64'])
        append += [mord_df]
    
    return pd.concat(append, axis=1)
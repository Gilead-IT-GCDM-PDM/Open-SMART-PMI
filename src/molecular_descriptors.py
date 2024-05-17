# -- STANDARD LIBRARY IMPORTS
import numpy as np
import pandas as pd

# -- CHEMICAL LIBRARY IMPORTS
from rdkit import Chem
from rdkit.Chem.AtomPairs import Torsions, Pairs
from mordred import Calculator, descriptors

# -- VARIABLES

# -- RDKit featurizers
descriptor_names = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())
get_descriptors = Chem.rdMolDescriptors.Properties(descriptor_names)

# Mordred featurizers
calc = Calculator(descriptors, ignore_3D=True)


# -- HELPER FUNCTIONS


def compute(smi_series: pd.Series) -> pd.DataFrame:
    """ 
    Compute molecular descriptors over a series of smiles strings.
    """
    dfs = []

    # -- use RDKit featurizers
    rdkit_desc = list(smi_series.apply(smi_to_descriptors))
    desc_df = pd.DataFrame(rdkit_desc, columns=descriptor_names)
    dfs += [desc_df]

    # -- remove features with nulls
    idx_to_exclude = desc_df[desc_df.isna().any(axis=1)].index

    # -- use Mordred featurizers
    mols = [Chem.MolFromSmiles(smi)
            for smi in smi_series if Chem.MolFromSmiles(smi) is not None]
    mord_df = calc.pandas(mols)
    dfs += [mord_df]
    # -- combine feature descriptors into df
    df = pd.concat(dfs, axis=1)

    if len(idx_to_exclude):
        print(f'... Removed the following SMILES {list(idx_to_exclude)}...')
        df = exclude_rows_from_df(df, idx_to_exclude)
        smi_series = exclude_rows_from_df(smi_series, idx_to_exclude)

    atom_pair_tuples = [atom_pairs(s) for s in smi_series]
    df['UNIQUETT'], df['UNIQUEAP'], df['CHIRAL_COUNT'] = zip(*atom_pair_tuples)
    df['CHIRAL_ALLATOM_RATIO'] = df.CHIRAL_COUNT / df.NumHeavyAtoms
    df['SMILES'] = smi_series
    return df


def exclude_rows_from_df(df, idx_to_exclude):
    return df[~df.index.isin(idx_to_exclude)]


def smi_to_descriptors(smile:str) -> np.array:
    '''
    Compute standard kit of rdkit descriptors
    '''
    # -- use RDKit featurizers
    mol = Chem.MolFromSmiles(smile, sanitize=True)
    if mol:
        return np.array(get_descriptors.ComputeProperties(mol))
    return []


def atom_pairs(smile:str) -> tuple:
    '''
    Calculate unique topological torsion and unique atom pairs on eligible features
    '''
    m = Chem.MolFromSmiles(smile)

    tt = Torsions.GetTopologicalTorsionFingerprint(m)
    unique_tt = len(tt.GetNonzeroElements())

    ap = Pairs.GetAtomPairFingerprint(m)
    unique_ap = len(ap.GetNonzeroElements())

    ccs = Chem.FindMolChiralCenters(m, includeUnassigned=True)

    return unique_tt, unique_ap, len(ccs)

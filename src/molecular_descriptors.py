# -- STANDARD LIBRARY IMPORTS
import numpy as np
import pandas as pd

# -- CHEMICAL LIBRARY IMPORTS
from rdkit import Chem
from rdkit.Chem.AtomPairs import Torsions, Pairs
from mordred import Calculator, descriptors

# -- VARIABLES

# RDKit featurizers
descriptor_names = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())
get_descriptors = Chem.rdMolDescriptors.Properties(descriptor_names)

# Mordred featurizers
calc = Calculator(descriptors, ignore_3D=True)


# -- HELPER FUNCTIONS


def compute(smi_series):
    """ Apply molecular descriptors to a series of smiles strings.
    return dataframe
    """
    df = []

    # -- use RDKit featurizers
    rdkit_desc = list(smi_series.apply(smi_to_descriptors))
    desc_df = pd.DataFrame(rdkit_desc, columns=descriptor_names)
    df += [desc_df]

    # remove features with nulls
    idx_to_exclude = desc_df[desc_df.isna().any(axis=1)].index

    # -- use Mordred featurizers
    mols = [Chem.MolFromSmiles(smi)
            for smi in smi_series if Chem.MolFromSmiles(smi) is not None]
    mord_df = calc.pandas(mols)
    df += [mord_df]

    # combine feature descriptors into df
    df = pd.concat(df, axis=1)
    trim = lambda df: df[~df.index.isin(idx_to_exclude)]
    print(f'... Removed the following SMILES {list(idx_to_exclude)}...')
    df = trim(df)

    # we can pass mol instead of smile: TODO
    # -- add in uniqueTT and uniqueAP
    atom_pair_tuples = [atom_pairs(s) for s in trim(smi_series)]
    df['UNIQUETT'], df['UNIQUEAP'], df['CHIRAL_COUNT'] = zip(*atom_pair_tuples)
    df['CHIRAL_ALLATOM_RATIO'] = df.CHIRAL_COUNT / df.NumHeavyAtoms

    df['SMILES'] = trim(smi_series)
    return df


def smi_to_descriptors(smile):
    """ Compute standard kit of rdkit descriptors
    """
    # -- use RDKit featurizers
    mol = Chem.MolFromSmiles(smile, sanitize=True)
    if mol:
        return np.array(get_descriptors.ComputeProperties(mol))
    return []


def atom_pairs(smile):
    """ Calculate uniqueTT and uniqueAP on eligible features
    """
    m = Chem.MolFromSmiles(smile)

    tt = Torsions.GetTopologicalTorsionFingerprint(m)
    unique_tt = len(tt.GetNonzeroElements())

    ap = Pairs.GetAtomPairFingerprint(m)
    unique_ap = len(ap.GetNonzeroElements())

    ccs = Chem.FindMolChiralCenters(m, includeUnassigned=True)

    return unique_tt, unique_ap, len(ccs)

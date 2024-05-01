
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

IPythonConsole.ipython_useSVG=True
IPythonConsole.molSize = (500, 500)

molfile = """
  Mrv2311 03262415422D          

  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 12 11 0 0 0
M  V30 BEGIN ATOM
M  V30 1 N -3.2103 4.3298 0 0
M  V30 2 C -3.2103 1.6625 0 0
M  V30 3 P -1.6703 1.6625 0 0
M  V30 4 O -7.0603 5.6635 0 0
M  V30 5 O -7.624 3.5598 0 0
M  V30 6 O -0.1303 1.6625 0 0
M  V30 7 O -1.6703 0.1225 0 0
M  V30 8 C -3.9803 2.9961 0 0
M  V30 9 P -6.2903 4.3298 0 0
M  V30 10 C -5.5203 2.9961 0 0
M  V30 11 O -4.9566 5.0998 0 0
M  V30 12 O -1.6703 3.2025 0 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 2 8
M  V30 2 1 8 1
M  V30 3 1 8 10
M  V30 4 1 9 4
M  V30 5 1 9 5
M  V30 6 1 9 10
M  V30 7 2 9 11
M  V30 8 1 2 3
M  V30 9 1 3 6
M  V30 10 1 3 7
M  V30 11 2 3 12
M  V30 END BOND
M  V30 END CTAB
M  END
"""


def example2():
    m = Chem.MolFromSmiles('NC(CP(=O)(O)O)CP(=O)(O)O')
    centers = Chem.FindMolChiralCenters(m, force=True, includeUnassigned=True, useLegacyImplementation=False)
    Draw.MolToFile(m, 'example2.png', size=(100, 100), fitImage=False, imageType='png')
    print(len(centers))


def example1():
    mol = Chem.MolFromMolBlock(molfile)

    centers = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True, useLegacyImplementation=False)
    print(len(centers))

    for cc in Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True, useLegacyImplementation=False):
        at = mol.GetAtomWithIdx(cc[0])
        at.SetProp("atomLabel", cc[1])

    print(mol)


if __name__ == '__main__':
    example2()

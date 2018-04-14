import argparse
from rdkit import Chem

import numpy as np

from grammar_model.alphabets import MoleculeAlphabet

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
args = parser.parse_args()

a = MoleculeAlphabet()

with open(args.data, 'r') as fhandle:
    mol_strs = fhandle.read().strip().split('\n')

print("Preprocessing, this takes a couple minutes")
clean_strs, clean_idxs = [], []

rejected = 0
for mol_str in mol_strs:
    mol = Chem.MolFromSmiles(mol_str)
    if mol is None:
        rejected += 1
        continue
    Chem.Kekulize(mol)
    kekule_str = Chem.MolToSmiles(mol, kekuleSmiles=True)

    try:
        idx = a.expr2idx(kekule_str)
        clean_strs.append(kekule_str)
        clean_idxs.append(idx)
    except ValueError:
        rejected += 1
        continue

max_len = np.max([len(idx) for idx in clean_idxs])
data = np.zeros((len(clean_idxs), max_len))

for i, source in enumerate(clean_idxs):
    data[i, :len(source)] = source

if "zinc" in args.data:
    np.savez_compressed(args.data.split('.')[0], train=data[:-5000], test=data[-5000:])
else:
    np.savez_compressed(args.data.split('.')[0], test=data)

print("File {} processed, {} rejected.".format(args.data, rejected))

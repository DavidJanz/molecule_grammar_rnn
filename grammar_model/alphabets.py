import re

import numpy as np
import torch

try:
    from rdkit import rdBase
    from rdkit import RDLogger

    rdBase.DisableLog('rdApp.error')
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    from rdkit import Chem
except ImportError:
    print("RDKit not found!")


class Alphabet:
    def __init__(self):
        if self.alphabet is not None:
            self.alphabet = [' '] + self.alphabet
            self.alphabet_array = np.array(self.alphabet)

    def __len__(self):
        if self.alphabet is not None:
            return len(self.alphabet)
        else:
            return 0

    def _validate(self, expr):
        if not isinstance(expr, str):
            raise TypeError("Expr not a string")
        return 0

    def validate(self, idx):
        if not isinstance(idx, (tuple, list, torch.LongTensor)):
            raise TypeError("idx of incorrect type.")
        if (isinstance(idx, torch.LongTensor) and len(idx.size()) > 1) or isinstance(idx, list):
            print([self.validate(i) for i in idx])
            print(type([self.validate(i) for i in idx]))
            print("----------------------------")
            return torch.FloatTensor([self.validate(i) for i in idx])
        # if expression has no length
        if idx[0] == 0:
            return False
        expr = self.idx2expr(idx)
        return self._validate(str.strip(expr))

    def idx2expr(self, idx):
        if not isinstance(idx, (tuple, list, torch.LongTensor)):
            raise TypeError("idx of incorrect type.")
        return ''.join([self.alphabet[i] for i in idx])

    def expr2idx(self, expr):
        """Encodes expr as an array of indices, with each index
        being the corresponding characters index in the alphabet."""
        if not isinstance(expr, str):
            raise TypeError("Expr not a string")
        return torch.FloatTensor([self.alphabet.index(c) for c in expr])

    def generate_perturbed(self, x, p=0.05):
        x_uni = torch.from_numpy(np.random.choice(range(len(self)),
                                                  size=x.size()))
        mask = torch.rand(*x.size()) > p
        x_uni[mask] = x[mask]
        return x_uni


int_re = re.compile(r'([0-9]+)')


def re_repl_int_float(match_obj):
    return match_obj.group(0) + '.0'


class PythonAlphabet(Alphabet):
    def __init__(self):
        _numbers = list('0123456789')
        _symbols = list('+-=()*/<>!%')

        alphabet = _numbers + _symbols
        self.alphabet = alphabet
        super().__init__(terminal=True)

    def _validate(self, expr):
        try:
            eval(re.sub(int_re, re_repl_int_float, expr))
        except KeyboardInterrupt as e:
            raise e
        except Exception:
            return False
        return True


class MoleculeAlphabet(Alphabet):
    def __init__(self):
        _alphabet_atoms = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S']
        _alphabet_rings = ['1', '2', '3', '4', '5', '6', '7', '8']
        _alphabet_symbols = ['#', '(', ')', '+', '-', '=', '[', ']']

        self.alphabet = _alphabet_atoms + _alphabet_rings + _alphabet_symbols

        super().__init__()

    def expr2idx(self, expr):
        """Encodes expr as an array of indices, with each index
        being the corresponding characters index in the alphabet."""
        if not isinstance(expr, str):
            raise TypeError("Expr not a string")
        idx = []
        i = 0
        while i < len(expr):
            if expr[i:i + 2] in self.alphabet:
                idx.append(self.alphabet.index(expr[i:i + 2]))
                i += 2
            else:
                idx.append(self.alphabet.index(expr[i]))
                i += 1
        return np.array(idx)

    def _validate(self, expr):
        expr = expr.split(' ')[0]
        mol = Chem.MolFromSmiles(expr)
        if mol is not None:
            return True
        return False

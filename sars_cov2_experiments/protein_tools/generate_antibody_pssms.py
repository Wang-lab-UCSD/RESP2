"""Builds a 'handcrafted' feature set."""
import sys
import numpy as np

sys.path.append("..")
from constants import seq_encoding_constants as seC



def main():
    property_dict = {"hydropathy":
            {"A":1.8, "R":-4.5,
        "N":-3.5, "D":-3.5, "C":2.5, "Q":-3.5,
        "E":-3.5, "G":-0.4, "H":-3.2, "I":4.5,
        "L":3.8, "K":-3.9, "M":1.9, "F":2.8,
        "P":-1.6, "S":-0.8, "T":-0.7, "W":-0.9,
        "Y":-1.3, "V":4.2},

            "molecular_weight":
        {"A":89.0, "R":174.2, "N":132.1,
        "D":133.1, "C":121.2, "Q":146.1, "E":147.1,
        "G":75.1, "H":155.2, "I":131.2, "L":131.2,
        "K":146.2, "M":149.2, "F":165.2, "P":115.1,
        "S":105.1, "T":119.1, "W":204.2, "Y":181.2,
        "V":117.1},

            "charge":
        {"A":0., "R":1., "N":0., "D":-1., "C":0.,
        "Q":0., "E":-1., "G":0., "H":0., "I":0.,
        "L":0., "K":1., "M":0., "F":0.,
        "P":0., "S":0., "T":0., "W":0.,
        "Y":0., "V":0.}
    }

    for _, prop_set in property_dict.items():
        max_val = max(prop_set.values())
        min_val = min(prop_set.values())

        for key, val in prop_set.items():
            prop_set[key] = (val - min_val) / (max_val - min_val)

    categories = {}
    for aa in ["A", "I", "L", "V", "M"]:
        categories[aa] = "aliphatic_nonpolar"
    for aa in ["F", "Y", "W"]:
        categories[aa] = "aromatic"
    for aa in ["S", "T", "N", "Q", "K", "R", "E", "D"]:
        categories[aa] = "aliphatic_polar"
    for aa in ["C", "P", "G", "H", "-"]:
        categories[aa] = "special"

    aas = seC.aas
    sim_mat = np.zeros((21,21))

    for i, aa1 in enumerate(aas):
        sim_mat[i,i] = 1.

        for j in range(i+1, len(aas)):
            aa2 = aas[j]
            if categories[aa2] in ("special", "aromatic") or \
                    categories[aa1] in ("special", "aromatic"):
                continue
            if categories[aa2] != categories[aa1]:
                continue

            aa_dist = 0

            for _, prop_set in property_dict.items():
                aa_dist += (prop_set[aa1] - prop_set[aa2])**2

            sim_mat[i,j] = (3 - aa_dist) / 4
            sim_mat[j,i] = (3 - aa_dist) / 4

    factored_mat = np.linalg.cholesky(sim_mat)




if __name__ == "__main__":
    main()

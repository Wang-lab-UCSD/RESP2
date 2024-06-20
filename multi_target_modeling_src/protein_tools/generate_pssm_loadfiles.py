"""Generates cholesky-factored distance matrices from PFAsum
PSSM matrices."""
import os
import numpy as np



def factor_pssm(raw_mat, offset = 0.0):
    """Takes the input PSSM matrix, converts it to
    a distance matrix, then Cholesky-factors it."""
    dist_mat = np.zeros(raw_mat.shape)
    for i in range(raw_mat.shape[0]):
        for j in range(raw_mat.shape[1]):
            max_self_sim = np.max([raw_mat[i,i], raw_mat[j,j]])
            dist_mat[i,j] = (max_self_sim - raw_mat[i,j]) / max_self_sim

    dist_mat = dist_mat.max() - dist_mat
    dist_mat.flat[::dist_mat.shape[0]+1] += offset
    final_mat = np.linalg.cholesky(dist_mat) / np.sqrt(2)
    return final_mat.clip(min=0)


def get_factored_pfasum_pssm(percent_homology,
        desired_aa_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']):
    """Builds a factored PFAsum matrix for a given input filepath.

    Args:
        percent_homology (int): The percent homology cutoff. Determines
            which PFASUM file to use.
        desired_aa_order (list): A list which indicates
            the order in which AAs SHOULD appear.
    """
    raw_mat, aas = get_raw_pssm_matrix(percent_homology)
    final_mat = factor_pssm(raw_mat)

    reordered_mat = np.zeros_like(final_mat)
    for id1, aa in enumerate(desired_aa_order):
        reordered_mat[id1,:] = final_mat[aas.index(aa),:]

    return reordered_mat, desired_aa_order


def generate_pfasum_dotprod_file(percent_homology, fhandle):
    final_mat, aas = get_factored_pfasum_pssm(percent_homology)

    fhandle.write("class PFASUM%s_standardized:\n\n"%(percent_homology))
    fhandle.write("    @staticmethod\n    def get_aas():\n")
    fhandle.write("        return %s\n\n"%aas)
    fhandle.write("    @staticmethod\n    def get_mat():\n")
    matstring = ",\n".join(["          %s"%final_mat[i,:].tolist() for i in range(21)])
    fhandle.write("        return np.asarray([%s])\n\n\n"%matstring)



def get_raw_pssm_matrix(percent_homology):
    """Loads the raw pssm matrix into memory."""
    start_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(start_dir, "pfasum_matrices"))
    with open(f"PFASUM{percent_homology}.mat", "r", encoding = "utf-8") as pfa_handle:
        lines = [line for line in pfa_handle if not line.startswith("#")]
    os.chdir(start_dir)

    #Cut off the last four amino acids (e.g. B, Z) which are basically
    #"unclear aa"
    lines = lines[:-4]
    #The first line and first column contain the aas. Cut off the
    #unused symbols (B, Z etc) and add a gap.
    aas = lines[0].strip().split()[:-4] + ["-"]
    mat_rows = [[float(z) for z in line.strip().split()[1:-4]] +
               [-4.0] for line in lines[1:]]
    mat_rows += [[-4.0 for i in range(20)] + [1.0]]
    raw_mat = np.asarray(mat_rows)
    return raw_mat, aas




def write_raw_pssm(percent_homology, fhandle):
    """Loads the raw PSSM and then writes it to the Python file for
    easy loading."""
    final_mat, aas = get_raw_pssm_matrix(percent_homology)

    fhandle.write("class PFASUM%s_raw:\n\n"%(percent_homology))
    fhandle.write("    @staticmethod\n    def get_aas():\n")
    fhandle.write("        return %s\n\n"%aas)
    fhandle.write("    @staticmethod\n    def get_mat():\n")
    matstring = ",\n".join(["          %s"%final_mat[i,:].tolist() for i in range(21)])
    fhandle.write("        return np.asarray([%s])\n\n\n"%matstring)


def generate_all_pssm_loadfiles():
    """Generates a python file containing loadable
    factored PSSM matrices for various target percent
    homologies."""
    fpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(fpath)
    with open("pfasum_matrices.py", "w+", encoding="utf-8") as fhandle:
        fhandle.write("import numpy as np\n\n\n")

        for homology in [95,90,85,75,62]:
            generate_pfasum_dotprod_file(homology, fhandle)

        write_raw_pssm(90, fhandle)


if __name__ == "__main__":
    generate_all_pssm_loadfiles()

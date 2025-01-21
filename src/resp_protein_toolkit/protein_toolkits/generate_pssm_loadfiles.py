"""A simple script to generates cholesky-factored distance matrices from PFAsum
PSSM matrices."""
import os
import numpy as np



def get_factored_pfasum_pssm(percent_homology, offset = 0.0):
    """Builds a factored PFAsum matrix for a given input filepath
    but does not sort it (caller must do this).

    Args:
        percent_homology (int): The percent homology cutoff. Determines
            which PFASUM file to use.
        desired_aa_order (list): A list which indicates
            the order in which AAs SHOULD appear.
    """
    raw_mat, aas = get_raw_pssm_matrix(percent_homology)

    dist_mat = np.zeros(raw_mat.shape)
    for i in range(raw_mat.shape[0]):
        for j in range(raw_mat.shape[1]):
            max_self_sim = np.max([raw_mat[i,i], raw_mat[j,j]])
            dist_mat[i,j] = (max_self_sim - raw_mat[i,j]) / max_self_sim

    dist_mat = dist_mat.max() - dist_mat
    dist_mat.flat[::dist_mat.shape[0]+1] += offset
    final_mat = np.linalg.cholesky(dist_mat) / np.sqrt(2)
    return final_mat.clip(min=0), aas

def get_pfasum_distmat(percent_homology, offset = 0.0):
    """Builds a distance matrix to use as a representation but
    does not sort it (caller must do this).

    Args:
        percent_homology (int): The percent homology cutoff. Determines
            which PFASUM file to use.
        desired_aa_order (list): A list which indicates
            the order in which AAs SHOULD appear.
    """
    raw_mat, aas = get_raw_pssm_matrix(percent_homology)

    dist_mat = np.zeros(raw_mat.shape)
    for i in range(raw_mat.shape[0]):
        for j in range(raw_mat.shape[1]):
            max_self_sim = np.max([raw_mat[i,i], raw_mat[j,j]])
            dist_mat[i,j] = (max_self_sim - raw_mat[i,j]) / max_self_sim

    return dist_mat, aas


def get_raw_pssm_matrix(percent_homology):
    """Loads the raw pssm matrix into memory without sorting."""
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

    # Sort the matrix so that amino acids are in alphabetical order
    # (extremely important).

    return raw_mat, aas



def aa_sort_matrix(input_mat, aas, desired_aa_order):
    """Sorts the rows of the input matrix so the aa
    corresponding to each row is in desired_aa_order.
    Extremely important that this is consistent."""
    reordered_mat = np.zeros_like(input_mat)
    for id1, aa in enumerate(desired_aa_order):
        reordered_mat[id1,:] = input_mat[aas.index(aa),:]

    return reordered_mat




def generate_all_pssm_loadfiles():
    """Generates a python file containing loadable
    factored PSSM matrices for various target percent
    homologies."""
    fpath = os.path.dirname(os.path.abspath(__file__))
    os.chdir(fpath)

    # This is the amino acids + gap in alphabetical order. EXTREMELY
    # important to use same order everywhere.
    desired_aa_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

    for homology in [95,90,85,75,62,30,20]:
        output_fname = f"PFASUM{homology}_standardized.npy"
        unsorted_mat, aas = get_factored_pfasum_pssm(homology)
        final_mat = aa_sort_matrix(unsorted_mat, aas, desired_aa_order)
        np.save(output_fname, final_mat)

        output_fname = f"PFASUM{homology}_distmat.npy"
        unsorted_mat, aas = get_pfasum_distmat(homology)
        final_mat = aa_sort_matrix(unsorted_mat, aas, desired_aa_order)
        np.save(output_fname, final_mat)

        output_fname = f"PFASUM{homology}_raw.npy"
        unsorted_mat, aas = get_raw_pssm_matrix(homology)
        final_mat = aa_sort_matrix(unsorted_mat, aas, desired_aa_order)
        np.save(output_fname, final_mat)

    for homology in [11]:
        output_fname = f"PFASUM{homology}_distmat.npy"
        unsorted_mat, aas = get_pfasum_distmat(homology)
        final_mat = aa_sort_matrix(unsorted_mat, aas, desired_aa_order)
        np.save(output_fname, final_mat)

        output_fname = f"PFASUM{homology}_raw.npy"
        unsorted_mat, aas = get_raw_pssm_matrix(homology)
        final_mat = aa_sort_matrix(unsorted_mat, aas, desired_aa_order)
        np.save(output_fname, final_mat)


if __name__ == "__main__":
    generate_all_pssm_loadfiles()

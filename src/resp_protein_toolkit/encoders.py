"""Contains Python wrapper code for the main encoders."""
import os
import numpy as np
from resp_protein_toolkit.resp_protein_toolkit_ext import get_max_length, onehot_flat_encode_list, onehot_3d_encode_list
from resp_protein_toolkit.resp_protein_toolkit_ext import subsmat_flat_encode_list, subsmat_3d_encode_list
from resp_protein_toolkit.resp_protein_toolkit_ext import integer_encode_list




class OneHotProteinEncoder():
    """Provides basic one-hot encoding."""

    def __init__(self, alphabet = "gapped"):
        """Class constructor.

        Args:
            alphabet (str): One of 'standard', 'gapped' or 'expanded'.
                'standard' means only the basic 20 aas are allowed. 'gapped'
                means gaps are also allowed. 'expanded' means unusual AAs
                or low-confidence assignments like U, O, X, J, Z etc. are
                allowed.
        """
        if alphabet == "standard":
            self.expanded_symbol_set = False
            self.add_gaps = False
            self.alphabet_size = 20
        elif alphabet == "gapped":
            self.expanded_symbol_set = False
            self.add_gaps = True
            self.alphabet_size = 21
        elif alphabet == "expanded":
            self.expanded_symbol_set = True
            self.add_gaps = True
            self.alphabet_size = 27
        else:
            raise RuntimeError("Unexpected alphabet supplied. Must be in ['standard', 'gapped', "
                        "'expanded'].")


    def encode(self, sequence_list, flatten_output_array = False,
            max_length = None):
        """One-hot encode and return a numpy array. If flattened, it is of
        shape N x (M * A), where A is the alphabet size (20 for standard amino
        acids only, 21 if gaps are included, 27 if using an expanded alphabet),
        M is the number of amino acids and N is the number of datapoints.
        Otherwise it is a 3d array of shape N x M x A.

        Args:
            sequence_list (list): A list of sequences.
            flatten_output_array (bool): If True, a 2d flattened array is
                returned, otherwise a 3d array as discussed above.
            max_length: Either None or an int. If None, the sequence length
                is determined based on the longest sequence present in the
                input list and sequences are zero-padded to that size if
                necessary. If an int, sequences are zero-padded to be the
                size of max_length (unless they already are that length).
                Note that specifying max_length and then passing in sequences
                longer than that will cause an exception.

        Returns:
            encoded_seqs (np.ndarray): A numpy array of shape N x (M * A)
                or shape N x M x A depending on flatten_output_array as
                discussed above.

        Raises:
            RuntimeError: An exception is raised if invalid input is supplied.
        """
        if max_length is None:
            seq_length = get_max_length(sequence_list, False)

            if seq_length == 0:
                raise RuntimeError("Invalid sequences supplied. Check the input settings you used.")
        else:
            seq_length = max_length

        if flatten_output_array:
            output_array = np.zeros((len(sequence_list), self.alphabet_size * seq_length),
                    dtype=np.uint8)
            err_code = onehot_flat_encode_list(sequence_list, output_array,
                    self.expanded_symbol_set, self.add_gaps)
        else:
            output_array = np.zeros((len(sequence_list), seq_length, self.alphabet_size),
                    dtype=np.uint8)
            err_code = onehot_3d_encode_list(sequence_list, output_array,
                    self.expanded_symbol_set, self.add_gaps)

        if err_code != 1:
            raise RuntimeError("Invalid sequences supplied. Check the input settings you used.")

        return output_array



class IntegerProteinEncoder():
    """Provides integer encoding."""

    def __init__(self, alphabet = "gapped"):
        """Class constructor.

        Args:
            alphabet (str): One of 'standard', 'gapped' or 'expanded'.
                'standard' means only the basic 20 aas are allowed. 'gapped'
                means gaps are also allowed. 'expanded' means unusual AAs
                or low-confidence assignments like U, O, X, J, Z etc. are
                allowed.
        """
        if alphabet == "standard":
            self.expanded_symbol_set = False
            self.add_gaps = False
            self.alphabet_size = 20
        elif alphabet == "gapped":
            self.expanded_symbol_set = False
            self.add_gaps = True
            self.alphabet_size = 21
        elif alphabet == "expanded":
            self.expanded_symbol_set = True
            self.add_gaps = True
            self.alphabet_size = 27
        else:
            raise RuntimeError("Unexpected alphabet supplied. Must be in ['standard', 'gapped', "
                        "'expanded'].")


    def encode(self, sequence_list, max_length = None):
        """One-hot encode and return a numpy array of shape N x M, where N is
        the number of datapoints and M is the number of amino acids.

        Args:
            sequence_list (list): A list of sequences.
            all_same_length (bool): If True, the sequences are expected to
                be all the same length; if they are not there is an exception.
                If False, sequences are zero-padded to be the same length.
            max_length: Either None or an int. If None, the sequence length
                is determined based on the longest sequence present in the
                input list and sequences are zero-padded to that size if
                necessary. If an int, sequences are zero-padded to be the
                size of max_length (unless they already are that length).
                Note that specifying max_length and then passing in sequences
                longer than that will cause an exception.

        Returns:
            encoded_seqs (np.ndarray): A numpy array of shape N x M (see above).

        Raises:
            RuntimeError: An exception is raised if invalid input is supplied.
        """
        if max_length is None:
            seq_length = get_max_length(sequence_list, False)

            if seq_length == 0:
                raise RuntimeError("Invalid sequences supplied. Check the input settings you used.")
        else:
            seq_length = max_length

        output_array = np.zeros((len(sequence_list), seq_length),
                    dtype=np.uint8)
        err_code = integer_encode_list(sequence_list, output_array,
                    self.expanded_symbol_set, self.add_gaps)

        if err_code != 1:
            raise RuntimeError("Invalid sequences supplied. Check the input settings you used.")

        return output_array



class SubstitutionMatrixEncoder():
    """Encodes input proteins using a substitution matrix."""

    def __init__(self, homology = "90", rep_type = "std"):
        """Class constructor.

        Args:
            submat (str): The homology level to use. Currently supported is
                95, 90, 85, 75, 62.
            rep_type (str): If 'dist', the substitution matrix is
                used to build a distance matrix, each row of which is used
                as a representation. If 'std', the distance matrix is then Cholesky-
                factored and scaled so that the squared Euclidean distance between
                the length-21 representation for any two aas (or gaps) will
                yield the substitution matrix distance. If 'raw', the raw row
                of the substitution matrix is used as a representation.

        Raises:
            RuntimeError: An exception is raised if an unsupported option is requested.
        """
        current_dir = os.getcwd()
        fpath = os.path.abspath(os.path.dirname(__file__))
        os.chdir(os.path.join(fpath, "protein_toolkits"))

        if rep_type == 'std':
            expected_file = f"PFASUM{homology}_standardized.npy"
        elif rep_type == "dist":
            expected_file = f"PFASUM{homology}_distmat.npy"
        elif rep_type == "raw":
            expected_file = f"PFASUM{homology}_raw.npy"
        else:
            raise RuntimeError("Rep type must be one of 'std', 'raw', 'dist'.")

        if expected_file not in os.listdir():
            raise RuntimeError("Unsupported options requested.")

        # IMPORTANT: It is assumed here that the rows of the loaded PSSM
        # correspond to the AAs in alphabetical order. This is enforced when
        # generating these matrices by the script in protein_toolkits.
        # Using or generating matrices not in such order may yield unexpected results.
        self.pssm_matrix = np.load(expected_file).astype(np.float32)
        os.chdir(current_dir)


    def encode(self, sequence_list, flatten_output_array = False,
            max_length = None):
        """Encode and return a numpy array. If flattened, it is of
        shape N x (M * 21), where M is the number of amino acids and
        N is the number of datapoints. Otherwise it is a 3d array of
        shape N x M x 21.

        Args:
            sequence_list (list): A list of sequences.
            flatten_output_array (bool): If True, a 2d flattened array is
                returned, otherwise a 3d array as discussed above.
            max_length: Either None or an int. If None, the sequence length
                is determined based on the longest sequence present in the
                input list and sequences are zero-padded to that size if
                necessary. If an int, sequences are zero-padded to be the
                size of max_length (unless they already are that length).
                Note that specifying max_length and then passing in sequences
                longer than that will cause an exception.

        Returns:
            encoded_seqs (np.ndarray): A numpy array of shape N x (M * 21)
                or shape N x M x 21 depending on flatten_output_array as
                discussed above.

        Raises:
            RuntimeError: An exception is raised if invalid input is supplied.
        """
        if max_length is None:
            seq_length = get_max_length(sequence_list, False)

            if seq_length == 0:
                raise RuntimeError("Invalid sequences supplied. Check the input settings you used.")
        else:
            seq_length = max_length

        if flatten_output_array:
            output_array = np.zeros((len(sequence_list), 21 * seq_length),
                    dtype=np.float32)
            err_code = subsmat_flat_encode_list(sequence_list, output_array,
                    self.pssm_matrix)
        else:
            output_array = np.zeros((len(sequence_list), seq_length, 21),
                    dtype=np.float32)
            err_code = subsmat_3d_encode_list(sequence_list, output_array,
                    self.pssm_matrix)

        if err_code != 1:
            raise RuntimeError("Invalid sequences supplied. Check the input settings you used.")

        return output_array

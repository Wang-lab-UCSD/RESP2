"""Tests basic functionality for one hot, integer and substitution
matrix with PFAsum encoding."""
import random
import unittest
import numpy as np
from resp_protein_toolkit import OneHotProteinEncoder, IntegerProteinEncoder, SubstitutionMatrixEncoder



class TestBasicEncoders(unittest.TestCase):


    def test_error_checking(self):
        """Check that sequences which have known issues are flagged
        as such, and that deliberately invalid inputs are recognized."""
        # Use tokenizers with default settings (these will accept gapped
        # sequences).
        tokenizers = [OneHotProteinEncoder(), IntegerProteinEncoder(),
                SubstitutionMatrixEncoder()]

        problematic_sequence_lists = [["AaGGGTTY", "AXGG--Y", "CZYYD"]]

        for tokenizer in tokenizers:
            for slist in problematic_sequence_lists:
                with self.assertRaises(RuntimeError):
                    _ = tokenizer.encode(slist)

        # Use tokenizers with standard alphabet only.
        tokenizers = [OneHotProteinEncoder("standard"), IntegerProteinEncoder("standard")]

        problematic_sequence_lists = [["AaGGGTTY", "A-GG--Y", "C-YYD", "ZTTYCD"]]

        for tokenizer in tokenizers:
            for slist in problematic_sequence_lists:
                with self.assertRaises(RuntimeError):
                    _ = tokenizer.encode(slist)

        # Use tokenizers with expanded alphabet only.
        tokenizers = [OneHotProteinEncoder("expanded"), IntegerProteinEncoder("expanded")]

        problematic_sequence_lists = [["A@GGGTTY", "A-GG-#Y", "C1YYD", "ZTTYcD"]]

        for tokenizer in tokenizers:
            for slist in problematic_sequence_lists:
                with self.assertRaises(RuntimeError):
                    _ = tokenizer.encode(slist)

        # Try passing variable length sequences with all same length set.
        tokenizers = [OneHotProteinEncoder(), IntegerProteinEncoder(),
                SubstitutionMatrixEncoder()]

        problematic_sequence_lists = [["ACDTEGHFFY", "ACDTEGHFF", "AAA"]]

        for tokenizer in tokenizers:
            for slist in problematic_sequence_lists:
                with self.assertRaises(RuntimeError):
                    _ = tokenizer.encode(slist, max_length = 2)


    def test_onehot_encoding(self):
        """Check the one-hot encoder for correctness."""
        max_length, nseqs = 1000, 1001

        for flatten in [True, False]:
            for slength in [None, max_length]:
                for alphabet in ["standard", "gapped", "expanded"]:
                    encoder = OneHotProteinEncoder(alphabet)
                    seqs = generate_random_sequences(nseqs, max_length,
                        alphabet)
                    gt_array = onehot_encode(seqs, flatten, alphabet)
                    comp_array = encoder.encode(seqs, flatten, slength)
                    self.assertTrue(np.allclose(gt_array, comp_array))


    def test_integer_encoding(self):
        """Check the integer encoder for correctness."""
        max_length, nseqs = 1000, 1001

        for slength in [None, max_length]:
            for alphabet in ["standard", "gapped", "expanded"]:
                encoder = IntegerProteinEncoder(alphabet)
                seqs = generate_random_sequences(nseqs, max_length,
                    alphabet)
                gt_array = integer_encode(seqs, alphabet)
                comp_array = encoder.encode(seqs, slength)
                self.assertTrue(np.allclose(gt_array, comp_array))


    def test_submat_encoding(self):
        """Check the substitution matrix encoder for correctness."""
        max_length, nseqs = 1000, 1001

        for slength in [None, max_length]:
            for flatten in [True, False]:
                encoder = SubstitutionMatrixEncoder()
                seqs = generate_random_sequences(nseqs, max_length,
                    "gapped")
                gt_array = pfasum_encode(seqs, flatten)
                comp_array = encoder.encode(seqs, flatten, slength)
                self.assertTrue(np.allclose(gt_array, comp_array))



def onehot_encode(sequences, flatten = False, alphabet = "standard"):
    """Encode the input sequences using an inefficient pure Python
    routine to cross-check the more efficient wrapped routine."""
    aas = get_alphabet(alphabet)
    aa_dict = {k:i for i, k in enumerate(aas)}

    max_length = max([len(s) for s in sequences])
    output_array = np.zeros((len(sequences), max_length,
        len(aas)), dtype=np.uint8)

    for i, sequence in enumerate(sequences):
        for j, letter in enumerate(sequence):
            output_array[i,j,aa_dict[letter]] = 1

    if flatten:
        output_array = output_array.reshape((output_array.shape[0],
            output_array.shape[1] * output_array.shape[2]))
    return output_array



def integer_encode(sequences, alphabet = "standard"):
    """Encode the input sequences using an inefficient pure Python
    routine to cross-check the more efficient wrapped routine."""
    aas = get_alphabet(alphabet)
    aa_dict = {k:i for i, k in enumerate(aas)}

    max_length = max([len(s) for s in sequences])
    output_array = np.zeros((len(sequences), max_length), dtype=np.uint8)

    for i, sequence in enumerate(sequences):
        for j, letter in enumerate(sequence):
            output_array[i,j] = aa_dict[letter]

    return output_array


def pfasum_encode(sequences, flatten = False):
    """Encode the input sequences using an inefficient pure Python
    routine to cross-check the more efficient wrapped routine."""
    aas = get_alphabet("gapped")
    submat_encoder = SubstitutionMatrixEncoder()

    aa_mat = submat_encoder.pssm_matrix.copy()

    aa_dict = {k:i for i, k in enumerate(aas)}

    max_length = max([len(s) for s in sequences])
    output_array = np.zeros((len(sequences), max_length,
        len(aas)), dtype=np.float32)

    for i, sequence in enumerate(sequences):
        for j, letter in enumerate(sequence):
            output_array[i,j,:] = aa_mat[aa_dict[letter],:]

    if flatten:
        output_array = output_array.reshape((output_array.shape[0],
            output_array.shape[1] * output_array.shape[2]))
    return output_array



def generate_random_sequences(num_requested, max_length, alphabet = "standard"):
    """Generate random sequences. Lengths are also randomized (to
    be < all_one_length) unless all_one_length is True."""
    aas = get_alphabet(alphabet)

    random.seed(123)

    sequence_lengths = [random.randint(1, max_length) for
                i in range(num_requested)]

    sequences = []

    for i in range(num_requested):
        sequence = [random.choice(aas) for k in
                range(sequence_lengths[i])]
        sequences.append("".join(sequence))

    return sequences



def get_alphabet(alphabet = "standard"):
    """Return an aa_list corresponding to the requested
    alphabet."""
    if alphabet == "standard":
        aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    elif alphabet == "gapped":
        aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
    else:
        aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-',
                'B', 'J', 'O', 'U', 'X', 'Z']

    return aas


if __name__ == "__main__":
    unittest.main()

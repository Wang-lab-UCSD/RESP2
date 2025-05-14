"""A toolkit of classes for encoding individual sequences."""
import numpy as np

import torch
import esm
# Currently commented out because the AbLang experiments are expensive
# without providing any significant benefit and thus are not part of the
# standard experiment set. If you would like to run those experiments,
# uncomment this line.
# import ablang

from ..constants import seq_encoding_constants
from ..protein_tools.pfasum_matrices import PFASUM90_standardized
from ..utilities.model_loader_utilities import load_autoencoder
from ..utilities.data_loader_utilities import gen_chothia_dict
from .variant_handling import get_variant_dict




class OneHotEncoder():
    """Generates one-hot encoding for sequences."""

    def __init__(self):
        self.aa_dict = {seq_encoding_constants.aas[i]:i for i
                in range(len(seq_encoding_constants.aas))}

    def encode(self, sequences):
        """Encodes the input sequences as a 3d array."""
        encoded_sequences = []
        for sequence in sequences:
            encoded_seq = np.zeros((len(sequence), 21), dtype = np.uint8)
            for i, letter in enumerate(list(sequence)):
                encoded_seq[i, self.aa_dict[letter]] = 1
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences

    def encode_variable_length(self, sequences):
        """Encodes the input sequences as a 3d array when
        sequences are variable in length."""
        encoded_sequences = []
        max_len = max([len(seq) for seq in sequences])

        for sequence in sequences:
            encoded_seq = np.zeros((max_len, 21), dtype = np.uint8)
            for i, letter in enumerate(list(sequence)):
                encoded_seq[i, self.aa_dict[letter]] = 1
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences


class PFAStandardEncoder():
    """Encodes a sequence using a PFASUM (similar to BLOSUM) matrix,
    without inserting any gaps. The matrix used here has been
    standardized and undergone a Cholesky decomposition."""
    def __init__(self):
        aas = PFASUM90_standardized.get_aas()
        self.aa_dict = {aas[i]:i for i
                in range(len(aas))}
        self.mat = PFASUM90_standardized.get_mat()

    def encode(self, sequences):
        """Encodes the input sequences as a 3d array."""
        encoded_sequences = []
        for sequence in sequences:
            encoded_seq = np.stack([self.mat[self.aa_dict[letter],:] for letter in sequence])
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences

    def encode_variable_length(self, sequences):
        """Encodes the input sequences as a 3d array when
        the input sequences have different lengths."""
        encoded_sequences = []
        max_len = max([len(seq) for seq in sequences])

        for sequence in sequences:
            encoded_seq = np.zeros((max_len, 21), dtype = np.float32)
            for i, letter in enumerate(list(sequence)):
                encoded_seq[i,:] = self.mat[self.aa_dict[letter],:]
            encoded_sequences.append(encoded_seq)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences



class AbLangEncoder():
    """Encodes antibody sequences only using AbLang."""

    def __init__(self, start_dir):
        self.heavy_ablang = ablang.pretrained("heavy", device="cuda")
        self.heavy_ablang.freeze()


    def encode(self, sequences):
        """Returns the FAIR-ESM reps corresponding to mutated
        positions in the antigen sequence."""
        rescodings = self.heavy_ablang(sequences, mode='rescoding')
        return np.stack(rescodings)



class AntigenRawESMEncoder():
    """Encodes the known antigen sequences using the FAIR-ESM
    embedding with no fastconv. Can be used for antigens only,
    not antibodies."""

    def __init__(self, start_dir, expanded_list = False):
        esm_model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
        esm_model.cuda()

        self.rep_dict = {}

        raw_data = [(var_name, sequence) for var_name, (sequence, _) in
                get_variant_dict(start_dir, expanded_list).items()]

        _, _, batch_tokens = batch_converter(raw_data)
        batch_tokens = batch_tokens.cuda()
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33].cpu()
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            seqrep = token_representations[i, 1:tokens_len-1]
            sequence_representations.append(seqrep.numpy())

        for i, (_, sequence) in enumerate(raw_data):
            self.rep_dict[sequence] = sequence_representations[i]


    def encode(self, sequences):
        """Returns the FAIR-ESM reps corresponding to mutated
        positions in the antigen sequence."""
        encoded_sequences = [self.rep_dict[sequence] for sequence in sequences]
        return np.stack(encoded_sequences)





class AutoencoderEncoder():
    """Encodes an input sequence (a string) using one-hot encoding,
    inserting gaps as needed to produce a Chothia-numbered sequence
    that contains all the positions used in the autoencoder training
    set from Parkinson et al 2023. This is only applicable for
    antibodies (not RBDs). The one-hot encoding is then run through
    the autoencoder from the same."""

    def __init__(self, start_dir):
        self.autoencoder = load_autoencoder(start_dir)
        self.chothia_map, self.unused_chothia_positions = gen_chothia_dict(start_dir)


    def encode(self, sequences):
        """Returns the autoencoder-generated representations of the list
        of input sequences.
        """
        encoded_sequence = np.zeros((len(sequences),
                len(seq_encoding_constants.chothia_list), 21),
                    dtype = np.uint8)

        for i, sequence in enumerate(sequences):
            for j, letter in enumerate(sequence):
                encoded_sequence[i, self.chothia_map[j],
                    seq_encoding_constants.aas.index(letter)] = 1
            for position in self.unused_chothia_positions:
                encoded_sequence[i, position, -1] = 1

        return self.autoencoder.extract_hidden_rep(torch.from_numpy(encoded_sequence).float()).numpy()



class PChemPropEncoder():
    """Encodes a sequence using four physicochemical properties (the
    properties from Grantham et al. and expected charge at neutral pH).
    These are a little arbitrary, but it's hard to come up with a list
    of pchem predictors that ISN'T a little arbitrary, and for deep
    learning (and even to some extent for simpler models) the specific
    descriptors do not matter terribly as long as there is sufficient
    info to differentiate AAs."""

    def __init__(self):
        self.compositionality = seq_encoding_constants.COMPOSITION
        self.volume = seq_encoding_constants.VOLUME
        self.polarity = seq_encoding_constants.POLARITY
        self.mat = {aa:np.array([self.compositionality[aa], self.volume[aa],
                                 self.polarity[aa]]) for aa in seq_encoding_constants.aas}


    def encode_variable_length(self, sequences):
        """Encodes the input sequences as a 3d array."""
        encoded_sequences = []
        max_len = max([len(seq) for seq in sequences])

        for sequence in sequences:
            features = np.zeros((max_len, 3), dtype = np.float32)
            for i, aa in enumerate(sequence):
                features[i,0] = self.compositionality[aa]
                features[i,1] = self.volume[aa]
                features[i,2] = self.polarity[aa]
            encoded_sequences.append(features)

        encoded_sequences = np.stack(encoded_sequences)
        return encoded_sequences

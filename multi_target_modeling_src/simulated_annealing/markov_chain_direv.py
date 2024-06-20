'''This class performs modified simulated annealing, where
sequences are generated through random modifications to the wild-type,
The "score" from regression is then contrasted with the score of the premodification
sequence and the modification is accepted with a probability determined
by the temperature. Since there are multiple possible antigen targets (and
hence scores) for each sequence, the sequence is scored against all antigens,
and the scores are combined in a way that ensures only sequences which
are generally beneficial have a high probability of acceptance.

The accepted sequences from the chain and their scores are saved. Running
several chains yields a set of sequences that can be merged for experimental
evaluation.'''

import random
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..data_encoding.seq_encoder_functions import AntigenRawESMEncoder, OneHotEncoder
from ..data_encoding.variant_handling import get_variant_dict
from ..constants import seq_encoding_constants
from ..constants import simul_annealing_constants
from ..utilities.model_loader_utilities import load_final_models


#If MAX_NUM_FAILURES, i.e. if this number of sequences are rejected in
#a row, we terminate annealing (to avoid getting stuck in an endless loop
#of proposing and rejecting sequences). Otherwise annealing is terminated when
#temperature reaches the predefined threshold.
MAX_NUM_FAILURES = 100



class MarkovChainDE():
    """This class contains attributes and methods needed to perform modified
    simulated annealing. Inputs include the final trained model, the
    autoencoder, the wild type sequence and the probability distribution
    (derived from abundance of mutations at each position in the training
    set). Use a random seed for reproducibility.

    Attributes:
        model_type (str): One of "xgp", "cnn" depending on the type of
            model we have used.
        high_model: A model for the 'high' bin data.
        super_model: A model for the 'super' bin data.
        antibody_encoder: Encodes antibody sequences.
        high_antigens (np.ndarray): A stack of the encoded high-bin antigens.
        super_antigens (np.ndarray): A stack of the encoded super-bin antigens.
        scores (list): The scores accumulated to date.
        seqs (list): The sequences tested to date.
        acceptance_rate (float): The fraction of proposals that were
            accepted. Useful as a diagnostic.
        prob_distro (np.ndarray): The probability of specific mutations at specific
            positions; allows user to specify what mutations are most likely if
            desired.
        positions_for_mutation (list): The positions we will mutate.
        seed (int): The random number seed.
    """
    def __init__(self, start_dir:str, prob_distro, seed:int = 123,
            positions_for_mutation = None, model_type = "xgp"):
        """Class constructor.

        Args:
            start_dir (str): The project filepath.
            prob_distro (np.ndarray): The probability of specific mutations at specific
                positions; allows user to specify what mutations are most likely if
                desired.
            seed (int): Seed for reproducibility.
            positions_for_mutation: Either None or a numpy array with positions for
                mutation. If None, a default prespecified list is used.
            model_type (str): One of "xgp", "cnn". The CNN takes 3d input arrays,
                while xgp takes 2d.
        """
        variant_dict = get_variant_dict(start_dir, expanded_list = True)

        self.high_model, self.super_model = load_final_models(start_dir)
        self.antibody_encoder = OneHotEncoder()


        if model_type == "xgp":
            antigen_encoder = AntigenRawESMEncoder(start_dir, True)
        else:
            antigen_encoder = OneHotEncoder()

        self.encoded_high = antigen_encoder.encode([variant_dict[var][0]
                for var in simul_annealing_constants.HIGH_ANTIGENS])
        self.encoded_super = antigen_encoder.encode([variant_dict[var][0]
                for var in simul_annealing_constants.SUPER_ANTIGENS])

        if model_type == "xgp":
            self.encoded_high = self.encoded_high.mean(axis=1)
            self.encoded_super = self.encoded_super.mean(axis=1)
        else:
            self.encoded_high = torch.from_numpy(self.encoded_high).cuda()
            self.encoded_super = torch.from_numpy(self.encoded_super).cuda()

        self.model_type = model_type
        self.scores = []
        self.variances = []
        self.seqs = []
        self.prob_distro = prob_distro
        self.acceptance_rate = 0

        if positions_for_mutation is not None:
            self.positions_for_mutation = positions_for_mutation
        else:
            self.positions_for_mutation = simul_annealing_constants.TARGET_POSITIONS

        self.seed = seed


    def encode_seq(self, seq):
        """Encodes a proposed mutant seq, then runs it through the fc encoder
        to generate a representation. Next, pairs this representation with
        the PFASUM encoding of each possible variant. Returns two arrays:
        one containing concatenated encodings for HIGH bin prediction,
        the other containing concatenated encodings for SUPER bin prediction."""
        trimmed_seq = seq[seq_encoding_constants.LEFT_TRIM_POINT:
                          seq_encoding_constants.RIGHT_TRIM_POINT]
        x = self.antibody_encoder.encode([trimmed_seq])
        if self.model_type == "xgp":
            x = x.reshape((1, x.shape[1] * x.shape[2]))
            x_high = np.zeros((self.encoded_high.shape[0], x.shape[1] +
                self.encoded_high.shape[1]))
            x_super = np.zeros((self.encoded_super.shape[0], x.shape[1] +
                self.encoded_super.shape[1]))
            x_high[:,:x.shape[1]] = x
            x_super[:,:x.shape[1]] = x
            x_high[:,x.shape[1]:] = self.encoded_high
            x_super[:,x.shape[1]:] = self.encoded_super
            return x_high, x_super

        x_high = np.tile(x, (self.encoded_high.shape[0], 1, 1))
        x_super = np.tile(x, (self.encoded_super.shape[0], 1, 1))
        return x_high, x_super



    def score_all_antigens(self, seq):
        """Scores a sequence against all antigens and also calculates variance.

        Args:
            seq (str): The sequence to score.

        Returns:
            high_scores (np.ndarray): The scores for each antigen (in same order
                as seq_encoding_constants.HIGH_ANTIGENS).
            super_scores (np.ndarray): The scores for each antigen (in same order
                as seq_encoding_constants.SUPER_ANTIGENS).
            high_var (np.ndarray): The variance on the prediction for each antigen
                (in the same order as seq_encoding_constants.HIGH_ANTIGENS).
            super_var (np.ndarray): The variance on the prediction for each antigen
                (in the same order as seq_encoding_constants.SUPER_ANTIGENS).
        """
        x_high, x_super = self.encode_seq(seq)
        if self.model_type == "xgp":
            x_high, x_super = self.encode_seq(seq)
            high_scores, high_var = self.high_model.predict(x_high, get_var = True)
            super_scores, super_var = self.super_model.predict(x_super, get_var = True)
            return high_scores, super_scores, high_var, super_var

        x_high, x_super = torch.from_numpy(x_high).cuda(), torch.from_numpy(x_super).cuda()
        high_scores = self.high_model(x_high.float(),
                self.encoded_high.float()).detach().cpu().numpy()
        super_scores = self.super_model(x_super.float(),
                self.encoded_super.float()).detach().cpu().numpy()
        return high_scores, super_scores, np.zeros((high_scores.shape[0])), \
                np.zeros((super_scores.shape[0]))



    def get_min_score(self, seq):
        """Finds the minimum score against any antigen for an input sequence."""
        x_high, x_super = self.encode_seq(seq)
        high_scores = self.high_model.predict(x_high, get_var = False)
        super_scores = self.super_model.predict(x_super, get_var =False)
        return min([ np.min(high_scores), np.min(super_scores) ])



    def get_concat_scores(self, seq):
        """Returns the antigen scores as a single array."""
        x_high, x_super = self.encode_seq(seq)
        high_scores = self.high_model.predict(x_high, get_var = False)
        super_scores = self.super_model.predict(x_super, get_var =False)
        return np.concatenate([high_scores, super_scores])



    def calc_transition_prob(self, proposed_seq, temperature,
            current_high_scores, current_super_scores):
        """Calculates the transition probability for a proposed sequence according to
        an update rule."""
        new_high_scores, new_super_scores, new_high_var, new_super_var = \
                self.score_all_antigens(proposed_seq)
        if new_high_var.max() > simul_annealing_constants.MAX_HIGH_VAR or \
                new_super_var.max() > simul_annealing_constants.MAX_SUPER_VAR:
            return -1, new_high_scores, new_super_scores, new_high_var, new_super_var

        high_probs = np.exp( ((new_high_scores - current_high_scores) /
            temperature).max().clip(max=0)  )
        super_probs = np.exp( ((new_super_scores - current_super_scores) /
            temperature).max().clip(max=0) )

        acceptance_prob = np.prod(high_probs) * np.prod(super_probs)

        return acceptance_prob, new_high_scores, new_super_scores, new_high_var, new_super_var


    def plot_scores(self):
        """Simple helper function for plotting scores from a completed chain."""
        fig, ax = plt.subplots(1)
        ax.plot(self.scores, linewidth=2.0)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Model score")
        ax.set_title("Scores for in silico directed evolution")
        return fig, ax


    def run_chain(self, max_iterations:int = 3000, seed:int = 123,
            cooldown:float = 0.995, sparse_mutations:int = -1):
        """This function runs a simulated annealing experiment as described above.

        Args:
            max_iterations (int): The maximum number of iterations for
                the chain.
            seed (int): The random seed.
            cooldown (float): The rate at which to cool down. Slower means more
                exploration of (possibly useless) sequences.
            sparse_mutations (int): If > 0, try to revert to WT any time
                there are more than this number of mutations. Encourages the
                search to stay as close to WT as possible.
        """
        num_accepted = 0
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        original_seq = list(seq_encoding_constants.wt)
        current_seq = list(seq_encoding_constants.wt)
        current_high_scores, current_super_scores, _, _ = \
                self.score_all_antigens(current_seq)

        i, temp, num_failures = 0, 25.0, 0

        while temp >= 0.01:
            chosen_one = np.random.choice(self.positions_for_mutation, size=1)[0]
            num_mutated_positions = len([mp for mp, wp in zip(current_seq, original_seq)
                    if mp != wp])
            if num_mutated_positions >= sparse_mutations > 0:
                new_aa = original_seq[chosen_one]
            else:
                new_aa = seq_encoding_constants.aas[np.random.choice(21, size=1,
                    p=self.prob_distro[chosen_one,:])[0]]

            #If we accidentally selected the existing aa, try again, without changing
            #temp or any other settings.
            if new_aa == current_seq[chosen_one]:
                continue


            proposed_seq = current_seq.copy()
            proposed_seq[chosen_one] = new_aa

            acceptance_prob, new_high_scores, new_super_scores, new_high_var, new_super_var = \
                    self.calc_transition_prob(proposed_seq, temp,
                    current_high_scores, current_super_scores)
            runif = np.random.uniform()

            if acceptance_prob > runif:
                current_seq = copy.copy(proposed_seq)
                current_high_scores = new_high_scores.copy()
                current_super_scores = new_super_scores.copy()

                num_accepted += 1
                num_failures = 0
            else:
                num_failures += 1

            self.scores.append(np.concatenate([current_high_scores, current_super_scores]))
            self.variances.append(np.concatenate([new_high_var, new_super_var]))
            self.seqs.append(''.join(current_seq))

            temp *= cooldown
            i += 1
            if i % 250 == 0:
                print(f"Temperature: {temp}")
            if i > max_iterations or num_failures > MAX_NUM_FAILURES:
                print(f"Num consecutive failures: {num_failures}")
                print(f"Num iterations: {i}")
                break

        self.scores = np.stack(self.scores)
        self.variances = np.stack(self.variances)
        self.acceptance_rate = num_accepted / max_iterations


    def polish(self, orig_mutant, thresh = 0.95):
        """This function is available as an option to do
        additional processing on sequences
        selected by the simulated annealing process. Ideally we would like
        to use a sequence with a small number of mutations from the original.
        The polish function takes each mutant and tries to return the amino acid
        at as many mutated positions as possible to the aa present in the wild
        type at that position, while keeping the score within a threshold of the
        score for the original selected mutant."""
        revised_mutant = list(orig_mutant)
        score_thresh = thresh * self.get_concat_scores(revised_mutant)
        keep_cycle = True

        while keep_cycle:
            idx = []
            score_tracker = []
            for position in self.positions_for_mutation:
                proposed_mut = revised_mutant.copy()
                if proposed_mut[position] != seq_encoding_constants.wt[position]:
                    proposed_mut[position] = seq_encoding_constants.wt[position]
                    score_tracker.append(self.get_concat_scores(proposed_mut) - score_thresh)
                    idx.append(position)

            score_tracker = np.stack(score_tracker)
            if np.max(np.min(score_tracker, axis=1)) < 0:
                keep_cycle = False
                break

            best_mut = idx[np.argmax(np.min(score_tracker, axis=1))]
            revised_mutant[best_mut] = seq_encoding_constants.wt[best_mut]

        high_scores, super_scores, high_var, super_var = self.score_all_antigens(revised_mutant)
        revised_mutant = ''.join(revised_mutant)
        scores = np.concatenate([high_scores, super_scores])
        var = np.concatenate([high_var, super_var])
        return revised_mutant, scores, var

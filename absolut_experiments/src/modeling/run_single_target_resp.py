"""Runs the in siico directed evolution phase of the RESP
pipeline on the input data, using the trained models saved to
the results folder. Unlike run_resp, this code finds an
antigen against a single target rather than trying to
optimize against multiple targets (which is a little
harder)."""
import os
import math
import random
import pickle
import numpy as np
from resp_protein_toolkit import SubstitutionMatrixEncoder, OneHotProteinEncoder
from resp_protein_toolkit import InSilicoDirectedEvolution as ISDE
from ..utilities.utilities import read_raw_data_files, load_saved_model
from ..constants import constants


class SingleTargetSequenceScorer:
    """This object houses the saved models and scores an input sequence when
    only one antigen is being targeted."""

    def __init__(self, project_dir, target_protein,
                 target_model):
        """Constructor.

        Args:
            project_dir (str): The home dir for the project.
            target_protein (str): The protein to target.
            target_model (str): The model to use.
        """
        self.target_model = target_model

        self.protein_code = target_protein
        self.model = load_saved_model(project_dir, target_protein,
                target_model)

        # We set an arbitrary high variance threshold and filter the candidates
        # after they are collected based on how their variance and score compare
        # to other candidates identified through search. The threshold shown
        # here will virtually never be exceeded in practice.
        self.variance_threshold = 50.

        self.encoder = SubstitutionMatrixEncoder(homology="62",
                                                     rep_type = "raw")



    def get_protein_code(self):
        """Convenience function for retrieving the protein codes."""
        return self.protein_code


    def get_variance_threshold(self):
        """Convenience function for obtaining the variance thresholds."""
        return self.variance_threshold


    def predict(self, sequence):
        """Assigns a score and corresponding uncertainties to
        an input sequence. VERY important to flip the sign on the
        score since RESP maximizes but negative scores indicate
        tighter binders.

        Note that at least one of the models we
        are evaluating does not calculate uncertainty and so will
        not return an uncertainty value; if we are using this model
        simply set uncertainty to 0. Finally, note that the vbnn
        has a relatively complicated encoding procedure which
        requires us to first insert gaps, since it can only
        handle fixed-length input."""

        if self.target_model == "xgpr":
            seqlen = np.array([len(sequence)], dtype=np.int32)
            enc_seq = self.encoder.encode([sequence],
                                max_length = constants.MAX_LENGTH)
            pred, var = self.model.predict(enc_seq, seqlen,
                    get_var=True)
        elif self.target_model == "CNN":
            enc_seq = self.encoder.encode([sequence],
                                max_length = constants.MAX_LENGTH)
            pred, var = self.model.predict(enc_seq), 0.
        else:
            enc_seq = self.encoder.encode([sequence],
                                max_length = constants.MAX_LENGTH)
            pred, var = self.model.predict(enc_seq, get_var=True)

        return -pred, var




def get_starting_sequences(project_dir, protein_code):
    """For each target, we have a number of different
    possible sequence lengths we could use to design
    a binder. We will arbitrarily seek binders of
    lengths 11,13,15,17 and 19. For each we will run
    4 chains with different random seeds and starting
    sequences to try to capture as much of the landscape
    as possible."""
    random.seed(123)

    desired_binder_lengths = [11,13,15,17,19]
    sequences_and_seeds = []
    min_scores, max_scores = [], []
    median_scores = []

    length_to_sequence = {d:[] for d in desired_binder_lengths}
    length_to_score = {d:[] for d in desired_binder_lengths}
    all_scores = []

    with open(os.path.join(project_dir, "absolut_data",
                    f"{protein_code}_500kNonMascotte.txt"), "r",
              encoding="utf-8") as fhandle:
        _ = fhandle.readline()
        _ = fhandle.readline()

        for line in fhandle:
            seq = line.split()[1]
            score = float(line.split()[4])
            all_scores.append(score)
            if len(seq) in length_to_sequence:
                length_to_sequence[len(seq)].append(seq)
                length_to_score[len(seq)].append(score)

    median_score = np.median(all_scores)
    min_scores.append(np.min(all_scores))
    max_scores.append(np.max(all_scores))
    median_scores.append(median_score)

    for i, desired_length in enumerate(desired_binder_lengths):
        seq_options = length_to_sequence[desired_length]
        assigned_scores = length_to_score[desired_length]
        # Use only weak binders to start. We use > here since
        # in this case a less negative (higher) value indicates
        # a weaker binder.
        seq_options = [seq for (seq, score) in zip(seq_options,
                    assigned_scores) if score > median_score]

        seq_choice = random.choice(seq_options)
        sequences_and_seeds.append( (seq_choice, i) )
        sequences_and_seeds.append( (seq_choice, i + 123) )

    return sequences_and_seeds, np.array(min_scores), np.array(max_scores), median_scores


def run_single_target_resp_search(project_dir, target_protein,
                    target_model):
    """Runs the RESP pipeline using ten randomly selected sequences
    with poor binding from the training data, each of a different
    length (so that we obtain possible sequences of different
    lengths)."""
    seq_scorer = SingleTargetSequenceScorer(project_dir, target_protein,
                        target_model)

    prot_code = seq_scorer.get_protein_code()
    # start_seqs is a list of tuples, each of which is a starting
    # sequence and random seed. Use each to run an ISDE chain.
    start_seqs, min_scores, max_scores, _ = \
            get_starting_sequences(project_dir, prot_code)

    accepted_seqs, accepted_scores, accepted_var = [], [], []
    cooldown = 0.99

    for i, (start_seq, seed) in enumerate(start_seqs):
        prob_distro = np.zeros((len(start_seq), 21))
        prob_distro[:,:-1] = 1/20

        isde = ISDE(seq_scorer,
                uncertainty_threshold=seq_scorer.get_variance_threshold(),
                seed = seed, prob_distro = prob_distro, approach="liberal")
        print(target_model)
        isde.run_chain(start_seq, cooldown=cooldown,
                       max_iterations=3000)

        accepted_seqs.append(isde.get_accepted_seqs())
        # Important to take negative here.
        scores = -np.array(isde.get_scores())

        accepted_scores.append(scores)
        accepted_var.append(isde.get_uncertainty())
        print(f"Completed {i+1} out of {len(start_seqs)} chains.")

    nametag = f"isde_single_target_{target_model}"

    with open(os.path.join(project_dir, "absolut_results",
            f"{prot_code}_{nametag}_output.pk"), "wb") as fhandle:
        pickle.dump({"start_seqs":start_seqs, "seqs":accepted_seqs,
                 "scores":accepted_scores, "var":accepted_var,
                 "worst_training_score":max_scores,
                 "best_training_score":min_scores}, fhandle)

    already_written_seqs = set()

    # Write the output in a format that Absolut recognizes, and
    # another format more useful for quick parsing. Filter for
    # low confidence unless using the CNN (which has no uncertainty).
    sequence_counter = 1

    all_variance = np.vstack(accepted_var)
    median_var = np.median(all_variance, axis=0)
    print(f"Median variance: {median_var}")

    with open(os.path.join(project_dir, "absolut_results",
            f"{prot_code}_{nametag}_selected.csv"), "w+",
              encoding="utf-8") as fhandle:
        fhandle.write("Sequence,score,variance\n")

        with open(os.path.join(project_dir, "absolut_results",
                f"{prot_code}_{nametag}_selected.rtxt"), "w+",
                encoding="utf-8") as txt_fhandle:

            for seq_group, score_group, variance_group in \
                    zip(accepted_seqs, accepted_scores, accepted_var):
                if score_group.shape[0] != len(seq_group):
                    raise RuntimeError("Nonmatching list lengths for ISDE output.")

                for i, seq in enumerate(seq_group):
                    if seq in already_written_seqs:
                        continue

                    # Remove any sequence where the 95% CI overlaps the best
                    # training sequence. This is fine for the CNN too, since
                    # it outputs variance of 0 for all predictions.
                    score_comp = score_group[i] + np.sqrt(variance_group[i]) * 1.96
                    score_is_valid = score_comp < min_scores
                    if score_is_valid.sum() < score_is_valid.shape[0]:
                        continue

                    if target_model == "CNN":
                        already_written_seqs.add(seq)
                        write_selected_seq_to_file(score_group, variance_group,
                                            sequence_counter, seq, i, fhandle,
                                                   txt_fhandle)
                        sequence_counter += 1
                        continue
                    # For non-CNN sequences, also make sure the variance is
                    # low. As a simple hack (also used in the RESP1 paper),
                    # we simply discard the sequences that have higher-than-
                    # median variance.
                    var_acceptable = variance_group[i] < median_var
                    if var_acceptable.sum() == var_acceptable.shape[0]:
                        already_written_seqs.add(seq)
                        write_selected_seq_to_file(score_group, variance_group,
                                            sequence_counter, seq, i, fhandle,
                                                       txt_fhandle)
                        sequence_counter += 1



def write_selected_seq_to_file(score_group, variance_group,
                sequence_counter, seq, idx, csv_handle, txt_handle):
    """Writes a selected sequence to file after first appropriately
    rounding the scores."""
    score1 = np.round(score_group[idx], 5)
    var1 = np.round(variance_group[idx], 5)
    csv_handle.write(f"{seq},{score1},{var1}\n")
    txt_handle.write(f"{sequence_counter}\t{seq}\n")

"""Runs the in siico directed evolution phase of the RESP
pipeline on the input data, using the trained models saved to
the results folder."""
import os
import math
import random
import pickle
import numpy as np
from resp_protein_toolkit import SubstitutionMatrixEncoder, OneHotProteinEncoder
from resp_protein_toolkit import InSilicoDirectedEvolution as ISDE
from ..utilities.utilities import read_raw_data_files, load_saved_model
from ..constants import constants


class SequenceScorer:
    """This object houses the saved models and scores an input sequence."""

    def __init__(self, project_dir, target_protein,
                 target_model):
        """Constructor.

        Args:
            project_dir (str): The home dir for the project.
            target_protein (str): The protein to target.
            target_model (str): The model to use.
        """
        if target_protein not in constants.TARGET_PROTEIN_GROUPS:
            raise RuntimeError("Unspecified target group supplied.")

        self.target_model = target_model

        self.protein_codes = constants.TARGET_PROTEIN_GROUPS[target_protein]
        self.models = [load_saved_model(project_dir, p, target_model) for p in
                self.protein_codes]

        # We set an arbitrary high variance threshold and filter the candidates
        # after they are collected based on how their variance and score compare
        # to other candidates identified through search. The threshold shown
        # here will virtually never be exceeded in practice.
        self.variance_thresholds = np.array([50 for p in self.protein_codes])

        self.encoder = SubstitutionMatrixEncoder(homology="62",
                                                     rep_type = "raw")



    def get_protein_codes(self):
        """Convenience function for retrieving the protein codes."""
        return self.protein_codes


    def get_variance_thresholds(self):
        """Convenience function for obtaining the variance thresholds."""
        return self.variance_thresholds


    def predict(self, sequence):
        """Assigns two scores and corresponding uncertainties to
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
            all_preds = [m.predict(enc_seq, seqlen, get_var=True)
                         for m in self.models]
        elif self.target_model == "CNN":
            enc_seq = self.encoder.encode([sequence],
                                max_length = constants.MAX_LENGTH)
            all_preds = [(m.predict(enc_seq), np.zeros((1)))
                         for m in self.models]
        else:
            enc_seq = self.encoder.encode([sequence],
                                max_length = constants.MAX_LENGTH)
            all_preds = [m.predict(enc_seq, get_var=True)
                         for m in self.models]

        preds = -np.concatenate([p[0] for p in all_preds])
        var = np.concatenate([p[1] for p in all_preds])
        return preds, var




def get_starting_sequences(project_dir, target_protein_group):
    """For each target, we have a number of different
    possible sequence lengths we could use to design
    a binder. We will arbitrarily seek binders of
    lengths 11,13,15,17 and 19. For each we will run
    4 chains with different random seeds and starting
    sequences to try to capture as much of the landscape
    as possible."""
    random.seed(123)

    if target_protein_group not in constants.TARGET_PROTEIN_GROUPS:
        raise RuntimeError("Invalid target protein group supplied.")

    desired_binder_lengths = [11,13,15,17,19]
    sequences_and_seeds = []
    min_scores, max_scores = [], []
    median_scores = []

    for protein_code in constants.TARGET_PROTEIN_GROUPS[target_protein_group]:
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


def run_resp_search(project_dir, target_protein_group,
                    target_model):
    """Runs the RESP pipeline using ten randomly selected sequences
    with poor binding from the training data, each of a different
    length (so that we obtain possible sequences of different
    lengths)."""
    seq_scorer = SequenceScorer(project_dir, target_protein_group,
                        target_model)

    prot_codes = seq_scorer.get_protein_codes()
    # start_seqs is a list of tuples, each of which is a starting
    # sequence and random seed. Use each to run an ISDE chain.
    start_seqs, min_scores, max_scores, _ = \
            get_starting_sequences(project_dir, target_protein_group)

    accepted_seqs, accepted_scores, accepted_var = [], [], []
    cooldown = 0.99

    for i, (start_seq, seed) in enumerate(start_seqs):
        prob_distro = np.zeros((len(start_seq), 21))
        prob_distro[:,:-1] = 1/20

        isde = ISDE(seq_scorer,
                uncertainty_threshold=seq_scorer.get_variance_thresholds(),
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

    nametag = f"isde_{target_model}"

    with open(os.path.join(project_dir, "absolut_results",
            f"{target_protein_group}_{nametag}_output.pk"), "wb") as fhandle:
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
            f"{target_protein_group}_{nametag}_selected.csv"), "w+",
              encoding="utf-8") as fhandle:
        fhandle.write(f"Sequence,{prot_codes[0]}_score,{prot_codes[1]}_score,"
                f"{prot_codes[0]}_variance,{prot_codes[1]}_variance\n")

        with open(os.path.join(project_dir, "absolut_results",
                f"{target_protein_group}_{nametag}_selected.rtxt"), "w+",
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
                    score_comp = score_group[i,:] + np.sqrt(variance_group[i,:]) * 1.96
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
                    var_acceptable = variance_group[i,:] < median_var
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
    score1 = np.round(score_group[idx,0], 5)
    score2 = np.round(score_group[idx,1], 5)
    var1 = np.round(variance_group[idx,0], 5)
    var2 = np.round(variance_group[idx,1], 5)
    csv_handle.write(f"{seq},{score1},{score2},{var1},{var2}\n")
    txt_handle.write(f"{sequence_counter}\t{seq}\n")




def find_start_sequences_external_use(project_dir, target_protein_group):
    """Harvest starting sequences for use by other pipelines. The
    starting sequences extracted here are the same ones used for
    RESP, but have to be set up differently since external pipelines
    must run against each antigen individually."""
    random.seed(123)

    if target_protein_group not in constants.TARGET_PROTEIN_GROUPS:
        raise RuntimeError("Invalid target protein group supplied.")

    desired_binder_lengths = [11,13,15,17,19]

    for protein_code in constants.TARGET_PROTEIN_GROUPS[target_protein_group]:
        with open(os.path.join(project_dir, f"{protein_code}_selected_seqs.txt"),
                "w+", encoding="utf-8") as output_handle:
            length_to_results = {d:[] for d in desired_binder_lengths}
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
                    if len(seq) in length_to_results:
                        length_to_results[len(seq)].append((seq, score))

            median_score = np.median(all_scores)

            for desired_length in desired_binder_lengths:
                # Use only weak binders to start. We use > here since
                # in this case a less negative (higher) value indicates
                # a weaker binder. Note that when writing to output we will
                # flip the sign on the score since external pipelines
                # maximize not minimize fitness.
                seq_options = [(seq, score) for (seq, score) in
                        length_to_results[desired_length] if score > median_score]

                for _ in range(4):
                    seq_choice = random.choice(seq_options)
                    output_handle.write(f"{seq_choice[0]},{seq_choice[1]}\n")

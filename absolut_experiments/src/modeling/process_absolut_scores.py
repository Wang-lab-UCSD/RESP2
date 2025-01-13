"""The RESP pipeline generates candidates, and we then score
these using Absolut. This final step checks the Absolut-assigned
scores to see how many of them are better than anything in the
training set (or better than most sequences in the training set)."""
import os
import numpy as np
from ..constants import constants
from ..utilities.utilities import read_raw_data_files


def check_absolut_scores(project_dir):
    """Checks the output absolut scores and compares
    them to the training set scores for each antigen.
    Loop over a predefined list of models and targets and
    check only those which are present."""
    final_scorepath = os.path.join(project_dir, "absolut_results",
                                   "absolut_scores")
    raw_datapath = os.path.join(project_dir, "absolut_data")

    configurations = ["cnn", "xgpr", "protein_mpnn",
                      "evodiff_80", "evodiff_90"]

    for target_group, target_list in constants.TARGET_PROTEIN_GROUPS.items():
        for configuration in configurations:
            training_set_scores = []
            output_scores = {}
            bad_file = False


            for i, protein_code in enumerate(target_list):
                score_fpath = os.path.join(final_scorepath,
                    f"{protein_code}FinalBindings_{configuration}.txt.gz")
                seqs, scores = read_raw_data_files([score_fpath])
                if scores is None:
                    bad_file = True
                    break
                # Weirdly, Absolut! sometimes outputs the same score for
                # two slides on the same sequence, resulting in a duplicate.
                # We fix this here.
                for seq, score in zip(seqs, scores):
                    if seq not in output_scores:
                        output_scores[seq] = np.zeros((len(target_list)))
                    output_scores[seq][i] = score

                score_fpath = os.path.join(raw_datapath,
                    f"{protein_code}_500kNonMascotte.txt")
                training_set_scores.append( read_raw_data_files([score_fpath])[1] )

            if bad_file:
                print("\nNot all expected output scores were present for "
                      f"{target_group}, {configuration}; this target + configuration "
                      "is skipped for now.")
                continue

            output_scores = np.stack([s for _, s in output_scores.items()])
            training_set_scores = np.array(training_set_scores).T


            # We will check 1) how many candidates were generated,
            # 2) what fraction were better than anything in the training set for both
            # antigens,
            # 3) what fraction were better than anything in the training set for at least
            # one antigen,
            # 4) bootstrapped 95% CI on the success rate.
            rng = np.random.default_rng(123)
            full_success, partial_success = get_success_rate(training_set_scores,
                                                         output_scores)

            bootstrapped_full, bootstrapped_partial = [], []
            for _ in range(1000):
                idx = rng.choice(output_scores.shape[0],
                                 size = output_scores.shape[0],
                                 replace = True)
                full, partial = get_success_rate(training_set_scores,
                                             output_scores[idx,:])
                bootstrapped_full.append(full)
                bootstrapped_partial.append(partial)

            bootstrapped_full = np.sort(bootstrapped_full)
            bootstrapped_partial = np.sort(bootstrapped_partial)
            print(f"\nFor {target_group}, {configuration}:\n"
                f"FULL SUCCESS RATE: {full_success}  "
                f"{bootstrapped_full[25]} to {bootstrapped_full[975]}\n"
                f"PARTIAL_SUCCESS_RATE: {partial_success}  "
                f"{bootstrapped_partial[25]} to {bootstrapped_partial[975]}\n"
                f"Generated {output_scores.shape[0]} candidates")



def get_success_rate(training_scores, output_scores):
    """Calculates the success rate for the output scores and training
    scores that are provided. A full success means the candidate is
    better against all antigens than the training data, a partial
    success means it is better against at least one antigen."""
    best_training_scores = np.min(training_scores, axis=0)
    successes = output_scores < best_training_scores[None,:]

    full_success = 100 * np.prod(successes, axis=1).sum() / successes.shape[0]
    partial_success = 100 * np.max(successes, axis=1).sum() / successes.shape[0]

    return np.round(full_success, 2), np.round(partial_success, 2)

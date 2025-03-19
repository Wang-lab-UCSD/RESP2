'''The functions here run a prespecified number of simulated annealing
chains and save the results, using markov_chain_direv.py.'''

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from antpack import SequenceScoringTool
from .markov_chain_direv import MarkovChainDE
from ..constants import seq_encoding_constants



def get_flat_prob_distro():
    """Generates a flat probability distribution from
    which to sample mutations. We set the probability of
    deletions to zero and the probability of cysteine
    mutations to zero -- we want neither to introduce
    gaps nor cysteines."""
    flat_prob_distro = np.ones((len(seq_encoding_constants.wt),21))
    flat_prob_distro[:,-1] = 0
    flat_prob_distro[:,seq_encoding_constants.aas.index("C")] = 0
    flat_prob_distro /= flat_prob_distro.sum(axis=1)[:,None]
    return flat_prob_distro



def build_mutation_list(sequence):
    """Generates a string indicating at what positions the input sequence
    differs from the wild type."""
    mutation_list = []
    for k, letter in enumerate(sequence):
        if letter != seq_encoding_constants.wt[k]:
            mutation_list.append(f"{seq_encoding_constants.wt[k]}{k+1}{letter}")
    return "_".join(mutation_list)



def run_annealing_chains(start_dir:str):
    """Runs a small number of simulated annealing chains and saves the results
    to file."""
    flat_prob_distro = get_flat_prob_distro()

    os.chdir(os.path.join(start_dir, "results_and_resources", "simulated_annealing"))
    if "simul_annealing_res.pk1" in os.listdir():
        print("Annealing results have already been saved.")
        os.chdir(start_dir)
        return

    marks = [MarkovChainDE(start_dir, flat_prob_distro) for i in range(5)]
    os.chdir(os.path.join("results_and_resources", "simulated_annealing"))


    for i, mark in enumerate(marks):
        mark.run_chain(3000, seed = i, sparse_mutations = 5)
        _, _ = mark.plot_scores()
        plt.savefig(f"Scores for chain {i}.png")
        plt.close()

    print("Now polishing sequences...")

    retained_seqs = set()
    score_list, var_list, seq_list = [], [], []

    for k, chain in enumerate(marks):
        score_indices = np.where(chain.scores.min(axis=1) > 3.5)[0]

        for idx in score_indices.tolist():
            seq = chain.seqs[idx]
            if seq in retained_seqs:
                continue

            high_scores, super_scores, high_var, super_var = \
                    chain.score_all_antigens(seq)
            scores = np.concatenate([high_scores, super_scores])
            var = np.concatenate([high_var, super_var])

            retained_seqs.add(seq)
            score_list.append(scores)
            var_list.append(var)
            seq_list.append(seq)

        print(f"Done with chain {k}")

    output_dict = {"seqs":seq_list, "scores":np.stack(score_list),
            "var":np.stack(var_list)} 
    with open("simul_annealing_res.pk1", "wb") as fhandle:
        pickle.dump(output_dict, fhandle)
    os.chdir(start_dir)



def analyze_annealing_results(start_dir):
    """This function analyzes the results of the simulated annealing experiments
    to find final sequences for experimental evaluation. We can only test 25
    sequences, so we need to find criteria to narrow down the list of
    initially selected sequences. We use the humanness of the suggested
    sequences, their scores and our uncertainty around their scores to
    narrow things down."""

    os.chdir(os.path.join(start_dir, "encoded_data"))
    raw_seqs = set(pd.read_csv("reorg_data.rtxt.gz").Sequence.tolist())

    os.chdir(os.path.join(start_dir, "results_and_resources", "simulated_annealing"))

    if "simul_annealing_res.pk1" not in os.listdir():
        raise ValueError("Simulated annealing not run yet.")

    with open("simul_annealing_res.pk1", "rb") as fhandle:
        mark_res = pickle.load(fhandle)

    seqs, full_scores, var = mark_res["seqs"], mark_res["scores"], mark_res["var"]
    scores = full_scores.min(axis=1)

    var = (var - var.min(axis=0)[None,:]) / (var.max(axis=0) - var.min(axis=0))[None,:]
    var = np.max(var, axis=1)

    humanness_score_tool = SequenceScoringTool(normalization="none")
    humanness_scores = humanness_score_tool.score_seqs(seqs)

    #Immediately remove any sequences with low (absolute or relative )humanness.
    #(See the AntPack paper for details.)
    idx = humanness_scores > -100
    idx *= humanness_scores > np.percentile(humanness_scores, 30)

    #Next, remove the sequences regarding which we have the least confidence.
    idx *= var < np.percentile(var, 60)

    #Finally, of the remaining sequences, take those with the highest scores. Take up to 29
    #(what we can experimentally test).
    cut_point = np.sort(scores[idx])[-29]
    idx *= scores >= cut_point


    scores, humanness_scores = scores[idx], humanness_scores[idx]
    seqs = [seqs[i] for i in range(len(idx)) if idx[i]]


    os.chdir(start_dir)
    os.chdir(os.path.join("results_and_resources", "selected_sequences"))

    output_dict = {"IDNum":np.arange(len(seqs)).tolist(), "Sequence":seqs,
            "lowest_score":np.round(scores, 2).tolist(),
            "humanness_score":np.round(humanness_scores, 2).tolist(),
            "mutations":[],
            "present_in_training_data":[],
            "selection_procedure":[]}

    for seq in seqs:
        output_dict["selection_procedure"].append("simulated_annealing")
        output_dict["mutations"].append(build_mutation_list(seq))

        if seq in raw_seqs:
            output_dict["present_in_training_data"].append("Yes")
        else:
            output_dict["present_in_training_data"].append("No")


    selected_seq_set = set(output_dict["Sequence"])
    if len(selected_seq_set) != len(output_dict["Sequence"]):
        raise ValueError("Duplication detected in simulated annealing output!")


    output_df = pd.DataFrame.from_dict(output_dict)
    output_df.to_csv("Selected_sequences.csv", index=False)

    print(f"Harvested {output_df.shape[0]} sequences.")



def id_most_important_positions(start_dir, num_to_retrieve = 6):
    """In this experiment, we select a subset of mutation locations by
    finding the ones the model indicates may be most useful in isolation.
    This yields a much smaller search space that is much easier to cover."""
    flat_prob_distro = get_flat_prob_distro()

    scoring_tool = MarkovChainDE(start_dir, flat_prob_distro)

    position_scores = []
    original_scores = scoring_tool.get_concat_scores(list(seq_encoding_constants.wt))

    for p in range(len(seq_encoding_constants.wt)):
        scores = []
        clone = list(seq_encoding_constants.wt)

        for aa in seq_encoding_constants.aas[:-1]:
            if aa == "C":
                continue
            clone[p] = aa
            score_shift = scoring_tool.get_concat_scores(clone) - original_scores
            scores.append(score_shift.min())

        position_scores.append(max(scores))

    top_positions = list(np.argsort(position_scores)[-num_to_retrieve:])
    print(np.sort(top_positions))

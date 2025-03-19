"""Reorganizes the assembled sequences to facilitate encoding."""
import os
import random
import numpy as np
import pandas as pd

from ..constants import seq_encoding_constants

def reorganize_data(start_dir):
    """Reorganizes the assembled sequence data into a single
    pair of train-test text files for easier access."""
    os.chdir(os.path.join(start_dir, "raw_data"))
    bin_descript = pd.read_csv("bin_description.csv")
    non_control_bins = bin_descript[bin_descript["Category"]!="control"].copy()
    non_control_bins.sort_values(by = "Bin", inplace = True)
    non_control_bins["Description"] = [f"{v1}_{v2}" for (v1, v2) in
                    zip(non_control_bins["Target"].tolist(),
                        non_control_bins["Category"].tolist())]

    os.chdir(os.path.join(start_dir, "encoded_data", "series2_data"))
    seq_dict = {}
    #This construction ensures that -- for this experiment -- the naive library
    #will be the last bin (since it is RH22).
    bin_map = {k : i for (i, k) in enumerate(non_control_bins["Bin"].tolist())}

    for bin_name in non_control_bins["Bin"].tolist():
        with open(f"{bin_name}.rtxt", "r", encoding="utf-8") as fhandle:
            for line in fhandle:
                seq, freq = line.strip().split(",")
                freq = int(freq)
                if seq in seq_dict:
                    if seq_dict[seq][bin_map[bin_name]] != 0:
                        raise ValueError("Repopulating field on "
                                f"{bin_name},{seq}")
                    seq_dict[seq][bin_map[bin_name]] = freq
                else:
                    seq_dict[seq] = np.zeros((len(bin_map)))
                    seq_dict[seq][bin_map[bin_name]] = freq
        print(f"{bin_name} complete.")

    WT = seq_encoding_constants.wt

    accepted_seqs, mutation_distro = [], []
    for seq, freq in seq_dict.items():
        #Remove very low frequency sequences.
        if freq.sum() < seq_encoding_constants.MIN_FREQ:
            continue
        if freq[-1] == 0:
            seq_dict[seq][-1] = 1
        accepted_seqs.append(seq)
        num_muts = [seq[k] for k in range(len(seq)) if seq[k] != WT[k]]
        mutation_distro.append(len(num_muts))

    print(f"On average, there are {np.mean(mutation_distro)} mutations per sequence.")

    #Create 10,000 out of sample seqs to verify that uncertainty behaves as expected.
    out_of_sample_seqs = []
    for i in range(10000):
        mutated_wt = list(WT)
        for k in range(8):
            mut_position = random.randint(seq_encoding_constants.LEFT_TRIM_POINT,
                    len(WT) + seq_encoding_constants.RIGHT_TRIM_POINT)
            mutated_wt[mut_position] = seq_encoding_constants.aas[random.randint(0,19)]
            if mutated_wt[mut_position] == WT[mut_position]:
                mutated_wt[mut_position] = seq_encoding_constants.aas[random.randint(0,19)]

        out_of_sample_seqs.append("".join(mutated_wt))

    accepted_seqs.sort()
    rng = np.random.default_rng(123)
    accepted_seqs = [accepted_seqs[i] for i in
            rng.permutation(len(accepted_seqs)).tolist()]

    print(f"Retained {len(accepted_seqs)} seqs.")

    os.chdir(os.path.join(start_dir, "encoded_data"))

    with open("out_of_sample_mutants.rtxt", "w+", encoding="utf-8") as fhandle:
        for out_of_sample_seq in out_of_sample_seqs:
            _ = fhandle.write(out_of_sample_seq)
            _ = fhandle.write("\n")

    write_reorg_data_to_file("reorg_data.rtxt", accepted_seqs,
                            seq_dict, non_control_bins)
    os.chdir(start_dir)
    print("Reorg complete.")




def write_reorg_data_to_file(fname, seq_list, seq_dict, non_control_bins):
    """Writes the reorganized data to file."""
    bin_name_string = ",".join(non_control_bins["Description"].tolist())
    with open(fname, "w+", encoding="utf-8") as fhandle:
        fhandle.write(f"Sequence,{bin_name_string}\n")
        for seq in seq_list:
            freqs = ",".join([str(int(f)) for f in seq_dict[seq].tolist()])
            _ = fhandle.write(f"{seq},{freqs}\n")

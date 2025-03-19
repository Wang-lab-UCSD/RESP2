"""Code for sequence prep. Not included in the core experiments
but can be re-run separately if desired."""
import os
import gzip

import numpy as np
import Bio
from Bio import SeqIO
from Bio.Seq import Seq

from ..constants import qual_params
from ..constants import seq_encoding_constants


def prep_all_sequences(start_dir, target_dir_list):
    """Preps all sequences from all 22 datasets for analysis."""
    for target_dir in target_dir_list:
        os.chdir(target_dir)
        left_file = [f for f in os.listdir() if
                f.endswith("fastq.gz") and "_R1_" in f]
        right_file = [f for f in os.listdir() if
                f.endswith("fastq.gz") and "_R2_" in f]

        if len(left_file) != 1 or len(right_file) != 1:
            raise RuntimeError("Found none / more than one "
                    f"left or right read in {target_dir}.")

        accepted_sequence_dict = {}

        dud_counter, net_seqs = 0, 0
        with gzip.open(left_file[0], "rt") as left_handle:
            with gzip.open(right_file[0], "rt") as right_handle:
                for left_seq, right_seq in zip(SeqIO.parse(left_handle, "fastq"),
                        SeqIO.parse(right_handle, "fastq")):
                    sequence, accepted = process_seq_pair(left_seq, right_seq)
                    net_seqs += 1
                    if net_seqs % 50000 == 0:
                        print(f"{net_seqs} processed.")
                    if accepted:
                        if sequence not in accepted_sequence_dict:
                            accepted_sequence_dict[sequence] = 0
                        accepted_sequence_dict[sequence] += 1
                    else:
                        dud_counter += 1

        os.chdir(start_dir)
        os.chdir("encoded_data")
        if "series2_data" not in os.listdir():
            os.mkdir("series2_data")
        os.chdir("series2_data")

        category_name = os.path.basename(target_dir).split("_")[0]
        accepted_unique_count, accepted_count = 0, 0
        with open(f"{category_name}.rtxt", "w+", encoding="utf-8") as fhandle:
            for key, value in accepted_sequence_dict.items():
                fhandle.write(f"{key},{value}\n")
                accepted_unique_count += 1
                accepted_count += value

        with open("processed_data_stats.txt", "a+", encoding="utf-8") as fhandle:
            fhandle.write(f"{category_name},{accepted_unique_count},"
                    f"{accepted_count},{dud_counter}\n")
        os.chdir(start_dir)
        print(f"{category_name} is complete.")


def process_seq_pair(left_seq, right_seq):
    """Processes a pair of raw reads in an effort to eliminate those that
    are low quality or have other issues."""
    accepted = True

    r_revcomp = str(right_seq.seq.reverse_complement()[:qual_params.right_offset])
    l_fwdcomp = str(left_seq.seq)

    rqual = right_seq._per_letter_annotations["phred_quality"][::-1][:qual_params.right_offset]
    lqual = left_seq._per_letter_annotations["phred_quality"]
    combined_seq = []
    if "N" in r_revcomp or "N" in l_fwdcomp:
        return None, False

    if np.min(lqual[9:-40]) < qual_params.phred_threshold or \
            np.min(rqual[80:]) < qual_params.phred_threshold:
        return None, False

    terminate_with_prejudice = False
    for i in range(qual_params.left_nonoverlapping):
        combined_seq.append(l_fwdcomp[i])

    for i in range(qual_params.left_nonoverlapping,
            qual_params.left_readlen):
        if l_fwdcomp[i] == r_revcomp[i-80]:
            combined_seq.append(l_fwdcomp[i])
        elif lqual[i] > qual_params.phred_threshold and lqual[i] > rqual[i-80]:
            combined_seq.append(l_fwdcomp[i])
        elif rqual[i-80] > qual_params.phred_threshold and rqual[i-80] > lqual[i]:
            combined_seq.append(r_revcomp[i-80])
        else:
            terminate_with_prejudice = True
            break

    for i in range(qual_params.left_offset,
            qual_params.left_readlen + qual_params.right_offset):
        combined_seq.append(r_revcomp[i])

    if terminate_with_prejudice:
        return None, False

    aa_seq = str(Seq("".join(combined_seq)).translate() )
    if "*" in aa_seq or aa_seq[:3] != "SAS" or aa_seq[-2:] != "GI":
        return None, False
    aa_seq = aa_seq[qual_params.LEFT_CUTOFF:qual_params.RIGHT_CUTOFF]

    if aa_seq[:seq_encoding_constants.LEFT_TRIM_POINT] != \
            seq_encoding_constants.wt[:seq_encoding_constants.LEFT_TRIM_POINT]:
        return None, False

    if aa_seq[seq_encoding_constants.RIGHT_TRIM_POINT:] != \
            seq_encoding_constants.wt[seq_encoding_constants.RIGHT_TRIM_POINT:]:
        return None, False
    return aa_seq, accepted

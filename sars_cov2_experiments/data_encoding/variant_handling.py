"""Contains code needed to process the variant (BA2, BA5, lambda etc).
sequence data to generate variant sequences for analysis."""
import os
from copy import deepcopy

from Bio import SeqIO

from ..constants import seq_encoding_constants



def generate_variant_lists(start_dir):
    """Constructs the variant sequences starting from the WT
    and lists of defining mutations for each.

    Args:
        start_dir (str): A filepath to the project directory.
    """
    os.chdir(os.path.join(start_dir, "results_and_resources",
                "variant_mutations"))
    with open("covid_wt_refseq.fasta", "r",
            encoding="utf-8") as fhandle:
        wt_refseq = list(SeqIO.parse(fhandle, "fasta"))[0]
    wt_refseq = str(wt_refseq.seq)

    variant_files = [f for f in os.listdir() if f.endswith("_mutations")]
    #Note that we want two variant lists: one for variants included in the
    #original set of experiments, and a second including additional strains
    #that may be of interest.
    expanded_variant_files = variant_files + [f for f in os.listdir() if
            f.endswith("_mutations_addtnl")]
    variant_files.sort()
    expanded_variant_files.sort()

    variants = construct_variant_sequences(variant_files, wt_refseq)
    with open("variant_sequences.rtxt", "w+",
            encoding="utf-8") as fhandle:
        for variant in variants:
            fhandle.write(f"{variant[0]},{variant[1]}\n")

    variants = construct_variant_sequences(expanded_variant_files, wt_refseq)
    with open("expanded_variant_sequences.rtxt", "w+",
            encoding="utf-8") as fhandle:
        for variant in variants:
            fhandle.write(f"{variant[0]},{variant[1]}\n")

    os.chdir(start_dir)


def construct_variant_sequences(variant_files, wt_refseq):
    """Builds a list of variant sequences by mutating the parent sequence at the
    positions indicated by the mutation list for each variant.

    Args:
        variant_files (list): A list of the variant files.
        wt_refseq (str): A string containing the parent sequence.

    Returns:
        variants (list): A list of tuples of form (name, sequence) sorted by name.
            Only the RBD portion of each sequence is included.
    """
    variants = [("wt", wt_refseq[seq_encoding_constants.RBD_CUTOFFS[0]:
                seq_encoding_constants.RBD_CUTOFFS[1]])]
    for variant_file in variant_files:
        variant_seq = list(deepcopy(wt_refseq))
        with open(variant_file, "r",
                encoding="utf-8") as fhandle:
            for line in fhandle:
                if not line.startswith("S:"):
                    raise ValueError(f"Line {line} on {variant_file} has "
                            "incorrect formatting.")

                mutation = line.strip().split(":")[1]
                start_aa, position, new_aa = mutation[0], int(mutation[1:-1]), \
                        mutation[-1]
                position -= 1
                if wt_refseq[position] != start_aa:
                    raise ValueError(f"{mutation} does not match wt refseq.")
                variant_seq[position] = new_aa

        variant_name = variant_file.split("_")[0].lower()
        variant_seq = "".join(variant_seq[seq_encoding_constants.RBD_CUTOFFS[0]:
                            seq_encoding_constants.RBD_CUTOFFS[1]])
        variants.append((variant_name, variant_seq))

    variants.sort(key=lambda x: x[0])
    return variants



def get_variant_dict(start_dir, expanded_list = False):
    """Gets a dictionary mapping variant name to sequence.

    Args:
        start_dir (str): A filepath to the project directory.
        expanded_list (bool): If True, get an expanded list of
            variants including xbb1. Otherwise, get only
            variants used in the original experiments.

    Returns:
        variant_dict (dict): A dict mapping variant name to a tuple of (full sequence,
            mutated positions only)."""
    os.chdir(os.path.join(start_dir, "results_and_resources", "variant_mutations"))

    if "variant_sequences.rtxt" not in os.listdir():
        raise ValueError("variant sequences not generated yet.")

    variant_dict = {}

    if expanded_list:
        with open("expanded_variant_sequences.rtxt", "r",
            encoding="utf-8") as fhandle:
            for line in fhandle:
                vname, vseq = line.strip().split(",")
                variant_dict[vname] = [vseq]
    else:
        with open("variant_sequences.rtxt", "r",
            encoding="utf-8") as fhandle:
            for line in fhandle:
                vname, vseq = line.strip().split(",")
                variant_dict[vname] = [vseq]


    vseqs = [vseq[0] for _, vseq in variant_dict.items()]
    variant_modified_positions = [i if len({vseq[i] for vseq in vseqs}) > 1
            else '-' for i in range(len(vseqs[0])) ]
    variant_modified_positions = [v for v in variant_modified_positions if v != '-']
    for _, subdict in variant_dict.items():
        mut_pos_only = [subdict[0][s] for s in variant_modified_positions]
        subdict.append("".join(mut_pos_only))

    os.chdir(start_dir)
    return variant_dict

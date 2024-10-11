"""Calculates the enrichment score for the input sequences for
each bin and writes this to file. This simplified code is
useful if only sequence and enrichment score information is
desired."""
import os
import gzip
import pickle
from .normalized_encode_sequences import get_out_of_sample_info, get_bin_counts
from statsmodels.stats.proportion import proportion_confint



def calc_enrichment_values(start_dir):
    """Generates the key encodings required for subsequent experiments."""
    os.chdir(os.path.join(start_dir, "encoded_data"))
    if "reorg_data.rtxt.gz" not in os.listdir():
        raise ValueError("Amino acid sequences not yet saved to appropriate location.")

    oos_dict = get_out_of_sample_info()
    if "encoded_seqs" not in os.listdir():
        os.mkdir("encoded_seqs")



    dset_dframes = get_bin_counts(start_dir)

    print("Encoding out of sample and key mutants...")
    #Encode a batch of sequences designed to be very different from the
    #rest of the dataset, to check that uncertainty-aware models correctly
    #flag these as "special". Also encode key mutants -- mutants that
    #we want to ensure are scored correctly when testing models.
    for key in ["high", "super"]:
        os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))
        write_group_to_file("out_of_sample_data", key, oos_dict[key])


    os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))

    #Loop over the train and test sets and high / super groups in
    #dset_dframes, which is organized as
    #{train/test : {high/super : {x:, y:, antigen: .
    for traintest_name, dset_dframe in dset_dframes.items():
        for label, data_dict in dset_dframe.items():
            os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))
            print(f"Working on {label}")

            write_group_to_file(traintest_name, label, data_dict)


def write_group_to_file(split_name, high_super_group, data_dict):
    """Writes all of the sequences in the input to a gzipped text file. The split
    name indicates whether the data is train or test, and the high_super_group
    indicates whether the data is high or super group."""

    output_fname = f"{split_name}_{high_super_group}_sequence_data.csv.gz"
    with gzip.open(output_fname, "wt") as fhandle:
        fhandle.write("Heavy_chain_sequence,Antigen_name,Enrichment_score\n")
        for seq, antigen, yvalue in zip(data_dict["x"], data_dict["antigen"],
                                             data_dict["y"]):
            fhandle.write(f"{seq},{antigen},{yvalue}\n")

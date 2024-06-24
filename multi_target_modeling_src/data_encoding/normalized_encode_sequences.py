"""Encodes assembled sequences using a variety of encoding schemes."""
import os
import gzip
import copy
import random
import pickle
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from ..utilities.model_loader_utilities import get_fc_encoder
from ..constants import seq_encoding_constants
from .variant_handling import get_variant_dict
from .seq_encoder_functions import OneHotEncoder
# Currently commented out because the AbLang experiments are expensive
# without providing any significant benefit and thus are not part of the
# standard experiment set. If you would like to run those experiments,
# uncomment this line.
#from .seq_encoder_functions import AbLangEncoder
from .seq_encoder_functions import PFAStandardEncoder
from .seq_encoder_functions import AntigenRawESMEncoder
from .seq_encoder_functions import AutoencoderEncoder


def get_raw_dataframe_keys():
    """Returns all of the column names that are useful for the dataframe
    containing the raw data, together with the group to which they belong
    (e.g. 'super' antigens, 'high' antigens etc."""
    superkeys = ['Beta_super', 'Omicron_super', 'BA5_super',
            'BA2_super', 'Gamma_super', 'Kappa_super']
    highkeys = ['Lambda_high', 'WT_high', 'Delta_high', 'Alpha_high']
    keys_of_interest = ["Sequence"] + superkeys + highkeys + ["naive_naive"]
    return superkeys, highkeys, keys_of_interest



def get_key_mutant_info():
    """Gets a list of key mutants (mutants where we have some prior
     knowledge regarding the likely status). We make predictions for
     these using the model as a sanity check."""
    _, _, keys_of_interest = get_raw_dataframe_keys()
    keys_of_interest = keys_of_interest[1:-1]
    key_mutant_dict = {key:{"x":[], "antigen":[], "y":[]} for key in ["high", "super"]}
    key_mutant_data = pd.read_csv("key_mutants.txt")

    #The key mutant data df has the column Sequence followed by all the high group
    #columns followed by the super group columns. Loop over all columns except the first
    #(sequence) and for each column, retrieve the antigen (part before _) and group
    #(high or super) (part after _). Add all sequences with their corresponding
    #data for THIS antigen to key_mutant_dict.
    for k in range(1, key_mutant_data.shape[1]):
        antigen, cat = key_mutant_data.columns[k].lower().split("_")
        key_mutant_dict[cat]["x"] += key_mutant_data.Sequence.tolist()
        key_mutant_dict[cat]["antigen"] += [antigen for j in range(key_mutant_data.shape[0])]
        key_mutant_dict[cat]["y"] += key_mutant_data.iloc[:,k].tolist()
    return key_mutant_dict


def get_out_of_sample_info():
    """Retrieves previously generated sequences with more mutations
    than are present in the average sequence in our dataset. This
    evaluates the ability of uncertainty-aware models to correctly
    express high uncertainty for sequences very different from the
    input data."""
    highkeys, superkeys, _ = get_raw_dataframe_keys()
    highkeys = [h.split("_", maxsplit=1)[0].lower() for h in highkeys]
    superkeys = [s.split("_", maxsplit=1)[0].lower() for s in superkeys]

    oos_dict = {"high":{}, "super":{}}
    random.seed(123)
    with gzip.open("out_of_sample_mutants.rtxt.gz", "rt",
            encoding = "utf-8") as fhandle:
        lines = [line.strip() for line in fhandle]
        for data_class, key_list in zip(["high", "super"],
                [highkeys, superkeys]):
            oos_dict[data_class]["x"] = lines
            #Randomly assign an antigen for each sequence.
            oos_dict[data_class]["antigen"] = \
                    [key_list[random.randint(0, len(key_list) - 1)]
                    for l in lines]
            #The yvalues here don't matter -- we use the oos mutants
            #to assess uncertainty quantitation only.
            oos_dict[data_class]["y"] = [0 for l in lines]
    return oos_dict




def get_bin_counts(start_dir):
    """Gets the total number of sequences in the reorganized data in each bin,
    then converts this to the probability of a given sequence in a bin and
    this in turn to an enrichment factor as compared to the naive library."""
    os.chdir(os.path.join(start_dir, "encoded_data"))

    _, _, keys_of_interest = get_raw_dataframe_keys()

    raw_data = pd.read_csv("reorg_data.rtxt.gz")[keys_of_interest].copy()
    output_dict = {key:{"x":[], "antigen":[], "y":[]}
            for key in ["high", "super"]}
    bin_sums = raw_data.iloc[:,1:].values.sum(axis=0)

    #This is the posterior predictive probability for each sequence in that antigen dataset
    #using binomial likelihood, beta prior with a = b = 0.5 (Jeffreys).
    probs = (raw_data.iloc[:,1:].values.astype(np.float64) + 0.5) / (bin_sums[None,:] + 1)

    #Loop over all columns excluding the first (sequence) and last (the naive
    #bin frequences). Exclude datapoints for which no frequency is > 1
    #or for which the 95% CI on naive bin prob overlaps the 95% CI on
    #target bin prob.
    naive_freqs, naive_probs = raw_data.iloc[:,-1].values, probs[:,-1]
    naive_ci_low, naive_ci_high = proportion_confint(raw_data.iloc[:,-1].values,
            raw_data.iloc[:,-1].values.sum(), method="jeffreys", alpha=0.25)

    for i, col in enumerate(raw_data.columns[1:-1]):
        raw_bin_freqs = raw_data[col].values
        retained_idx = (raw_bin_freqs > 1) | (naive_freqs > 1)
        ci_low, ci_high = proportion_confint(raw_data[col],
                raw_data[col].values.sum(), method="jeffreys", alpha=0.25)
        
        retained_idx *= (naive_probs > ci_high) | (naive_probs < ci_low)
        retained_idx *= (probs[:,i] > naive_ci_high) | (probs[:,i] < naive_ci_low)

        antigen, cat = col.split("_")
        output_dict[cat]["x"] += raw_data["Sequence"].values[retained_idx].tolist()
        output_dict[cat]["antigen"] += [antigen.lower() for j in range(retained_idx.sum())]
        enrich = np.log(probs[:,i][retained_idx]
                / naive_probs[retained_idx]).clip(min=0).tolist()
        output_dict[cat]["y"] += enrich

    output_dfs = {"train":{"high":{}, "super":{}},
            "test":{"high":{}, "super":{}}}
    rng = np.random.default_rng(123)

    for key in ["high", "super"]:
        idx = rng.permutation(len(output_dict[key]["x"])).tolist()
        cutoff = int(0.8 * len(idx))

        for dtype in ["x", "antigen", "y"]:
            output_dict[key][dtype] = [output_dict[key][dtype][i] for i in idx]
            output_dfs["train"][key][dtype] = output_dict[key][dtype][:cutoff]
            output_dfs["test"][key][dtype] = output_dict[key][dtype][cutoff:]

    return output_dfs



def generate_basic_encodings(start_dir):
    """Generates the key encodings required for subsequent experiments."""
    os.chdir(os.path.join(start_dir, "encoded_data"))
    if "reorg_data.rtxt.gz" not in os.listdir():
        raise ValueError("Amino acid sequences not yet saved to appropriate location.")

    oos_dict = get_out_of_sample_info()
    key_mutant_dict = get_key_mutant_info()

    if "encoded_seqs" not in os.listdir():
        os.mkdir("encoded_seqs")

    encoders = {"onehot":OneHotEncoder(),
                # Currently commented out because the AbLang experiments are expensive
                # without providing any significant benefit and thus are not part of the
                # standard experiment set. If you would like to run those experiments,
                # uncomment this line.
                # "ablang":AbLangEncoder(start_dir),
                "esm":AntigenRawESMEncoder(start_dir, True),
                "pfastd":PFAStandardEncoder(),
                "autoencoder":AutoencoderEncoder(start_dir),
                }


    dset_dframes = get_bin_counts(start_dir)
    variant_dict = get_variant_dict(start_dir, expanded_list = True)

    print("Encoding out of sample and key mutants...")
    #Encode a batch of sequences designed to be very different from the
    #rest of the dataset, to check that uncertainty-aware models correctly
    #flag these as "special". Also encode key mutants -- mutants that
    #we want to ensure are scored correctly when testing models.
    for key in ["high", "super"]:
        os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))
        encode_batches("train", f"out_of_sample_{key}",
                oos_dict[key], variant_dict, encoders)
        os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))
        encode_batches("train", f"key_mutants_{key}",
                key_mutant_dict[key], variant_dict, encoders)


    os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))

    #Loop over the train and test sets and high / super groups in
    #dset_dframes, which is organized as
    #{train/test : {high/super : {x:, y:, antigen: .
    for traintest_name, dset_dframe in dset_dframes.items():
        for label, data_dict in dset_dframe.items():
            os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))
            print(f"Working on {label}")

            encode_batches(traintest_name, label, data_dict, variant_dict, encoders)

            os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs"))
            with open(f"{label}_{traintest_name}_sequences.pk", "wb") as fhandle:
                pickle.dump(data_dict, fhandle)



def encode_batches(tt_split, label, stored_encodings, variant_dict,
                   encoders):
    """Encodes a batch of data using a variety of different encodings for later experiments.

    Args:
        tt_split (str): one of 'train', 'test', or some other folder name for other groups
            of interest. Determines which subdirectory this is saved to.
        label (str): One of 'high', 'super'. These are Group A and Group B antigens. Determines
            which directory this is saved to. For 'super', i.e. Group B, we only encode the
            data one way, since all model comparison experiments are done with Group A.
        stored_encodings (dict): A dict where key 'x' corresponds to a list of sequences
            and key 'antigen' corresponds to a list of corresponding antigens for each datapoint.
        variant_dict (dict): A dict mapping a variant name to [0] the full antigen sequence and
            [1] just the positions in the antigen sequence that are not constant.
        encoders (dict): A dict mapping encoding type names to encoders.
    """
    if label not in os.listdir():
        os.mkdir(label)
    os.chdir(label)
    if tt_split not in os.listdir():
        os.mkdir(tt_split.lower())
    os.chdir(tt_split)

    # Create an xGPR feature converter for various inputs so we
    # can also experiment with the xGPR two-stage convolution kernel
    # (the one-stage convolution kernel is more difficult to use
    # here). The onehot_converter can also be used for PFASum.
    ablang_converter = get_fc_encoder(seq_width = 768)
    onehot_converter = get_fc_encoder(seq_width = 21)
    autoenc_converter = get_fc_encoder(seq_width = 3)


    def flatten(arr):
        return arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))

    counter, batch_size = 0, 200
    for i in range(0, len(stored_encodings["x"]), batch_size):
        print(f"Saving seqs {i}:{i+batch_size}")
        #The N-terminal and C-terminal are excluded since we decided
        #not to modify these in this experiment, hence the trim points.
        untruncated_seqs = copy.deepcopy(stored_encodings["x"][i:i+batch_size])
        xseqs = [s[seq_encoding_constants.LEFT_TRIM_POINT:
            seq_encoding_constants.RIGHT_TRIM_POINT] for s in
            stored_encodings["x"][i:i+batch_size]]
        if len(xseqs) == 0:
            continue

        np.save(f"enrich_{counter}_y.npy",
                stored_encodings["y"][i:i+batch_size])

        #Either save the full antigen, or just the positions we care about for now.
        antigen = [variant_dict[a][1] for a in stored_encodings["antigen"][i:i+batch_size]]
        antigen_full = [variant_dict[a][0] for a in stored_encodings["antigen"][i:i+batch_size]]

        # For all groups except high, we need only the encoding that we
        # used to build the final model. For the high sequences, by
        # contrast, we will test a large number of models, therefore
        # we need many encodings (see below).
        if "high" not in label:
            x_pfa = encoders["pfastd"].encode(xseqs)
            a_esm = encoders["esm"].encode(antigen_full)
            a_pfa = encoders["pfastd"].encode(antigen)

            a_esm = encoders["esm"].encode(antigen_full)

            x_onehot = encoders["onehot"].encode(xseqs)
            a_onehot_partial = encoders["onehot"].encode(antigen)
            np.save(f"onehot_{counter}_concat.npy", np.hstack([flatten(x_onehot),
                                  flatten(a_onehot_partial)]).astype(np.float32) )
            np.save(f"onehotESM_{counter}_concat.npy", np.concatenate([flatten(x_onehot),
                                  a_esm.mean(axis=1)], axis=1).astype(np.float32) )
            counter += 1
            continue

        # Now save all of the different combinations that may be
        # of interest for our experiments when testing on high
        # (antigen group A).

        # One hot, either flat, conv1d extracted, or separate 3d arrays.

        x_onehot = encoders["onehot"].encode(xseqs)
        a_onehot_partial = encoders["onehot"].encode(antigen)
        a_onehot_full = encoders["onehot"].encode(antigen_full)
        a_esm = encoders["esm"].encode(antigen_full)


        x_lengths = np.array([x_onehot.shape[1] for _ in range(x_onehot.shape[0])],
                             dtype=np.int32)
        a_lengths = np.array([a_onehot_full.shape[1] for _ in range(a_onehot_full.shape[0])],
                             dtype=np.int32)

        x_conv = onehot_converter.predict(x_onehot, sequence_lengths=x_lengths,
                                          chunk_size=2000)
        a_conv = onehot_converter.predict(a_onehot_full, sequence_lengths=a_lengths,
                                          chunk_size=2000)

        np.save(f"onehot_{counter}_concat.npy", np.hstack([flatten(x_onehot),
                                  flatten(a_onehot_partial)]).astype(np.float32) )
        np.save(f"onehotconv_{counter}_concat.npy", np.hstack([x_conv, a_conv]).astype(np.float32) )
        np.save(f"onehotconvESM_{counter}_concat.npy", np.hstack([x_conv,
                                  a_esm.mean(axis=1)]).astype(np.float32) )
        np.save(f"onehotESM_{counter}_concat.npy", np.concatenate([flatten(x_onehot),
                                  a_esm.mean(axis=1)], axis=1).astype(np.float32) )
        np.save(f"onehot_{counter}_x.npy", x_onehot.astype(np.float32))
        np.save(f"onehot_{counter}_antigen.npy", a_onehot_full.astype(np.float32))


        # PFASUM, either flat, conv1d extracted, or separate 3d arrays.
        x_pfa = encoders["pfastd"].encode(xseqs)
        a_pfa = encoders["pfastd"].encode(antigen)
        x_lengths = np.array([x_pfa.shape[1] for _ in range(x_pfa.shape[0])],
                             dtype=np.int32)
        a_lengths = np.array([a_pfa.shape[1] for _ in range(a_pfa.shape[0])],
                             dtype=np.int32)
        x_conv = onehot_converter.predict(x_pfa, sequence_lengths=x_lengths,
                                          chunk_size=2000)
        a_conv = onehot_converter.predict(a_pfa, sequence_lengths=a_lengths,
                                          chunk_size=2000)

        np.save(f"pfaconv_{counter}_concat.npy", np.concatenate([x_conv, a_conv],
                                                        axis=1).astype(np.float32) )
        np.save(f"pfa_{counter}_concat.npy", np.concatenate([flatten(x_pfa),
                                  flatten(a_pfa)], axis=1).astype(np.float32) )
        np.save(f"pfa_{counter}_x.npy", x_pfa.astype(np.float32) )
        np.save(f"pfa_{counter}_antigen.npy", a_pfa.astype(np.float32) )


        x_autoencoder = encoders["autoencoder"].encode(untruncated_seqs)
        x_autoencoder = x_autoencoder[:,seq_encoding_constants.LEFT_TRIM_POINT:
                         seq_encoding_constants.RIGHT_TRIM_POINT,:]
        x_lengths = np.array([x_autoencoder.shape[1] for _ in range(x_autoencoder.shape[0])],
                             dtype=np.int32)
        x_conv = autoenc_converter.predict(x_autoencoder, sequence_lengths=x_lengths,
                                           chunk_size=2000)

        np.save(f"autoencoderconv_{counter}_concat.npy", np.hstack([x_conv,
                                  flatten(a_onehot_partial)]).astype(np.float32) )
        np.save(f"autoencoder_{counter}_concat.npy", np.hstack([flatten(x_autoencoder),
                                  flatten(a_onehot_partial)]).astype(np.float32) )
        np.save(f"autoencoder_{counter}_x.npy", x_autoencoder.astype(np.float32) )


        # AbLang (antibody only) or ESM (antigen only). This code is commented out
        # because the embedding experiments are expensive, requiring considerable
        #disk space and time. If you are interested in running these experiments,
        # uncomment this code.
        #x_ablang = encoders["ablang"].encode(xseqs)
        #x_conv = ablang_converter.predict(x_ablang, chunk_size=2000)

        #x_ablang_flat = flatten(x_ablang)
        #a_esm_flat = a_esm.mean(axis=1)

        #np.save(f"ablangconv_{counter}_concat.npy", np.hstack([x_conv,
        #                           a_esm.mean(axis=1)]).astype(np.float32) )
        #np.save(f"ablangESM_{counter}_concat.npy", np.hstack([x_ablang_flat,
        #                          a_esm_flat]).astype(np.float32) )
        #np.save(f"AbLang_{counter}_x.npy", x_ablang.astype(np.float32))
        #np.save(f"ESM_{counter}_antigen.npy", a_esm.astype(np.float32))

        counter += 1

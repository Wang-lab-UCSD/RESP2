"""Contains code for encoding input data in a format suitable for
the xGPR modeling library."""
import os
import math
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from resp_protein_toolkit import SubstitutionMatrixEncoder
from ..utilities.utilities import read_raw_data_files
from ..constants import constants



def data_encoding(project_dir):
    """Encodes the input data and saves it as numpy files in
    a subdirectory of the project dir."""
    target_groups = []
    for _, target_list in constants.TARGET_PROTEIN_GROUPS.items():
        target_groups += target_list

    os.chdir(project_dir)
    if "absolut_encoded_data" in os.listdir():
        shutil.rmtree("absolut_encoded_data")

    for target_group in target_groups:
        for data_type in ["train", "test", "val"]:
            os.makedirs(os.path.join(project_dir, "absolut_encoded_data",
                target_group, data_type), exist_ok = True)

    os.chdir(os.path.join(project_dir, "absolut_data"))


    for target_group in target_groups:
        x_all, y_all = read_raw_data_files([f"{target_group}_500kNonMascotte.txt"])

        # Just to be sure, check that all sequences are unique. (They SHOULD
        # be...but let's make sure.)
        found_x, retained_x, retained_y = set(), [], []
        for xseq, yval in zip(x_all, y_all):
            if xseq in found_x:
                continue
            retained_x.append(xseq)
            retained_y.append(yval)

        del x_all, y_all
        print(f"{len(retained_x)} sequences retained for subgroup {target_group}")

        x_tv, x_test, y_tv, y_test = train_test_split(retained_x,
                retained_y, test_size=0.2, random_state=123, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv,
                shuffle=True, test_size=0.2, random_state=123)

        encode_grp_data(x_train, y_train, constants.MAX_LENGTH,
                os.path.join(project_dir, "absolut_encoded_data",
                    target_group, "train"))
        encode_grp_data(x_test, y_test, constants.MAX_LENGTH,
                os.path.join(project_dir, "absolut_encoded_data",
                    target_group, "test"))
        encode_grp_data(x_val, y_val, constants.MAX_LENGTH,
                os.path.join(project_dir, "absolut_encoded_data",
                    target_group, "val"))

    os.chdir(project_dir)




def encode_grp_data(x, y, maxlen, output_path, batch_size = 2000):
    """Encodes the input x sequences, y values and antigen
    0-1 labels, saving the encoded arrays to disk."""
    subenc = SubstitutionMatrixEncoder(homology="62", rep_type = "raw")
    fcounter = 0

    for i in range(0, len(x), batch_size):
        sbatch = [len(s) for s in x[i:i+batch_size]]
        xseqs = x[i:i+batch_size]

        # First, encode for xGPR consumption.
        sbatch = np.array(sbatch).astype(np.int32)
        xbatch = subenc.encode(xseqs, flatten_output_array=False,
                               max_length = constants.MAX_LENGTH)
        ybatch = np.array(y[i:i+batch_size])

        np.save(os.path.join(output_path, f"{fcounter}_y.npy"), ybatch)
        np.save(os.path.join(output_path, f"{fcounter}_x.npy"),
                xbatch.astype(np.float32))
        np.save(os.path.join(output_path, f"{fcounter}_s.npy"), sbatch)

        # Next, encode for CNN consumption. This is the same as the xGPR
        # data but (in case we do decide to use a different encoding for the
        # CNN) is currently saved separately.
        xbatch = subenc.encode(xseqs, max_length = constants.MAX_LENGTH)
        np.save(os.path.join(output_path, f"{fcounter}_x_cnn.npy"),
                xbatch.astype(np.float32))

        # Next, encode using essentially IMGT numbering so that all sequences
        # are the same length. This makes the data suitable for use with a
        # fully connected vBNN.
        xgapped = []

        for seq in xseqs:
            ngaps = maxlen - len(seq)
            cutpoint = math.floor(len(seq) / 2)
            xgap = seq[:cutpoint] + "-" * ngaps + seq[cutpoint:]
            xgapped.append(xgap)

        xbatch = subenc.encode(xgapped, flatten_output_array = True)
        np.save(os.path.join(output_path, f"{fcounter}_x_vbnn.npy"), xbatch)

        fcounter += 1

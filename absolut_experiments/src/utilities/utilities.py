"""Shared utility functions for loading input data."""
import os
import gzip
import pickle


def read_raw_data_files(target_fpaths):
    """Reads a list of raw data files into memory, extracting
    only useful information. Assumes the data is in
    the format used by the Absolut database."""
    xseqs, yvalues = [], []

    for target_fpath in target_fpaths:
        if not os.path.exists(target_fpath):
            return None, None

        if target_fpath.endswith(".gz"):
            with gzip.open(target_fpath, "rt") as fhandle:
                # Skip first two lines, which are headers in the absolut data
                # format.
                _ = fhandle.readline()
                _ = fhandle.readline()

                for line in fhandle:
                    elements = line.split()
                    if elements[2] != "true":
                        continue

                    xseqs.append(elements[1])
                    yvalues.append(float(elements[4]))

        else:
            with open(target_fpath, "r", encoding="utf-8") as fhandle:
                # Skip first two lines, which are headers in the absolut data
                # format.
                _ = fhandle.readline()
                _ = fhandle.readline()

                for line in fhandle:
                    elements = line.split()
                    if elements[2] != "true":
                        continue

                    xseqs.append(elements[1])
                    yvalues.append(float(elements[4]))

    return xseqs, yvalues


def get_data_lists(project_dir, data_group_name, xsuffix = "x.npy"):
    """Gets lists of data files for xGPR. Returns
    train, validation and test as separate lists."""
    os.chdir(os.path.join(project_dir, "absolut_encoded_data",
        data_group_name))
    output_file_dict = {dtype:{"x":[], "y":[], "s":[]} for dtype in
            ["train", "val", "test"]}
    for dtype in output_file_dict.keys():
        os.chdir(dtype)
        xfiles = [f for f in os.listdir() if f.endswith(xsuffix)]
        xfiles = sorted(xfiles, key=lambda x: int(x.split("_")[0]))
        xfiles = [os.path.abspath(x) for x in xfiles]
        output_file_dict[dtype]["y"] = [x.replace(xsuffix, "y.npy")
                for x in xfiles]
        output_file_dict[dtype]["s"] = [x.replace(xsuffix, "s.npy")
            for x in xfiles]

        output_file_dict[dtype]["x"] = xfiles
        os.chdir("..")

    os.chdir(project_dir)
    return output_file_dict


def load_saved_model(project_dir, target_protein, target_model):
    """Loads the saved final models."""
    os.chdir(os.path.join(project_dir, "absolut_results"))
    if f"{target_protein}_{target_model}_final_model.pk" not in os.listdir():
        raise RuntimeError("Final models not yet saved to results "
                           "folder.")

    with open(f"{target_protein}_{target_model}_final_model.pk", "rb") as fhandle:
        loaded_model = pickle.load(fhandle)

    os.chdir(project_dir)

    return loaded_model

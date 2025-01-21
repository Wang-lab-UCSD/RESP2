"""Contains resources for loading saved models."""
import os
from ..constants import seq_encoding_constants



def get_traintest_flists(start_dir, model_class = "all",
        prefix = "pfasum", suffix = "x"):
    """Loads a list of training files and test files for
    a specified encoding type.

    Args:
        start_dir (str): File path to project dir.
        model_class (str): One of 'high',
            'super'. Indicates which data group under
            'encoded_seqs' to use.
        prefix (str): One of 'AbLang', 'ablangESM',
            'ablangESM_conv', 'ESM', 'onehot',
            'onehot_conv', 'pfa', 'pfa_conv'. All
            the different ways the data has been encoded.
        suffix (str): One of 'concat', 'x'. All except
            for the 3d arrays used by the non-FCCN
            DL models are concat.

    Returns:
        train_xfiles (list): A list of x array files.
        train_yfiles (list): A list of y array files.
        train_antigen_files (list): A list of antigen array files.
            Empty list if suffix is 'concat'.
        test_xfiles (list): A list of x array files.
        test_yfiles (list): A list of y array files.
        test_antigen_files (list): A list of antigen array files.
            Empty list if suffix is 'concat'.
    """
    os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs",
        model_class))
    if "train" not in os.listdir() or "test" not in os.listdir():
        raise ValueError("Data not encoded yet.")

    os.chdir("train")
    train_xfiles, train_yfiles, train_antigen_files = \
            get_flist(prefix, suffix)

    if len(train_xfiles) == 0:
        raise ValueError("Data not encoded yet.")
    os.chdir(os.path.join("..", "test"))
    test_xfiles, test_yfiles, test_antigen_files = \
            get_flist(prefix, suffix)

    os.chdir(start_dir)
    return train_xfiles, train_yfiles, train_antigen_files,\
            test_xfiles, test_yfiles, test_antigen_files





def get_trainvalid_flists(start_dir, model_class = "all",
        prefix = "pfasum", suffix = "x"):
    """Get a list of training files and validation files
    for a specified encoding type. Since sequences are
    shuffled pre-encoding, we can just split the file
    list to get train / validation.

    Args:
        start_dir (str): File path to project dir.
        model_class (str): One of 'all', 'High-super',
            'Medium-high'. Indicates which data group under
            'encoded_seqs' to use.
        prefix (str): One of 'AbLang', 'ablangESM',
            'ablangESM_conv', 'ESM', 'onehot',
            'onehot_conv', 'pfa', 'pfa_conv'. All
            the different ways the data has been encoded.
        suffix (str): One of 'concat', 'x'. All except
            for the 3d arrays used by the non-FCCN
            DL models are concat.

    Returns:
        train_xfiles (list): A list of x array files.
        train_yfiles (list): A list of y array files.
        train_antigen_files (list): A list of antigen array files.
            Empty list if suffix is 'concat'.
        valid_xfiles (list): A list of x array files.
        valid_yfiles (list): A list of y array files.
        valid_antigen_files (list): A list of antigen array files.
            Empty list if suffix is 'concat'.
    """
    os.chdir(os.path.join(start_dir, "encoded_data", "encoded_seqs",
        model_class))
    if "train" not in os.listdir() or "test" not in os.listdir():
        raise ValueError("Data not encoded yet.")

    os.chdir("train")
    train_xfiles, train_yfiles, train_antigen_files = \
            get_flist(prefix, suffix)

    if len(train_xfiles) == 0:
        raise ValueError("Data not encoded yet.")

    cutoff = int(0.8 * len(train_xfiles))
    valid_xfiles = train_xfiles[cutoff:]
    valid_yfiles = train_yfiles[cutoff:]
    valid_antigen_files = train_antigen_files[cutoff:]

    train_xfiles = train_xfiles[:cutoff]
    train_yfiles = train_yfiles[:cutoff]
    train_antigen_files = train_antigen_files[:cutoff]

    os.chdir(start_dir)
    return train_xfiles, train_yfiles, train_antigen_files,\
            valid_xfiles, valid_yfiles, valid_antigen_files




def get_flist(prefix = "pfasum", suffix = "x"):
    """Gets a list of files in the CURRENT working directory.

    Args:
        prefix (str): One of 'AbLang', 'ablangESM',
            'ablangESM_conv', 'ESM', 'onehot',
            'onehot_conv', 'pfa', 'pfa_conv'. All
            the different ways the data has been encoded.
        suffix (str): One of 'concat', 'x'. All except
            for the 3d arrays used by the non-FCCN
            DL models are concat.

    Returns:
        xfiles (list): A list of x array files.
        yfiles (list): A list of y array files.
        antigen_files (list): A list of antigen array files.
            Empty list if suffix is 'concat'.
    """
    f_all = [f for f in os.listdir() if f.endswith(".npy")]
    xfiles = [f for f in f_all if f.endswith(f"{suffix}.npy")
            and prefix == "_".join(f.split("_")[:-2])]
    kcodes = [x.split(f"{prefix}_")[1].split(f"_{suffix}.npy")[0]
              for x in xfiles]
    yfiles = [f"enrich_{kcode}_y.npy" for kcode in kcodes]

    antigen_files = []

    if suffix == "x":
        #For AbLang, we use the ESM embedding for the antigen since
        #AbLang is antibody only.
        if prefix == "AbLang":
            antigen_files = [f"ESM_{kcode}_antigen.npy"
                         for kcode in kcodes]
        #For the autoencoder, we use onehot encoding for antigen
        #only, since the autoencoder is antigen only.
        elif prefix == "autoencoder":
            antigen_files = [f"onehot_{kcode}_antigen.npy" for
                             kcode in kcodes]
        else:
            antigen_files = [f"{prefix}_{kcode}_antigen.npy"
                         for kcode in kcodes]

    xfiles.sort(key=lambda x: int(x.split("_")[-2]))
    yfiles.sort(key=lambda x: int(x.split("_")[-2]))
    antigen_files.sort(key=lambda x: int(x.split("_")[-2]))

    xfiles = [os.path.abspath(f) for f in xfiles]
    yfiles = [os.path.abspath(f) for f in yfiles]
    antigen_files = [os.path.abspath(f) for f in antigen_files]
    if len(xfiles) == 0:
        raise ValueError(f"No xfiles found using {prefix}, {suffix}.")
    if len(xfiles) != len(yfiles):
        raise ValueError(f"Different numbers of files using {prefix}, {suffix}.")
    return xfiles, yfiles, antigen_files





def gen_chothia_dict(start_dir):
    """This function generates dictionaries used to map input sequences to Chothia
    numbering, so that blanks can be inserted in appropriate places.

    Args:
        start_dir (str): A filepath to the starting directory.

    Returns:
        position_dict (dict): A dictionary that maps each position in a
            PDL1 sequence to the corresponding Chothia numbered position.
        unused_positions (list): A list of the Chothia numbered positions
            that are not used for PDL1.
    """
    os.chdir(start_dir)
    os.chdir("encoded_data")
    position_dict, positions_used = {}, set()
    positions_used = set()

    with open("chothia_template.rtxt", "r", encoding="utf-8") as input_file:
        template_codes = input_file.readlines()[0].strip().split(",")[13:135]

    expected_positions = seq_encoding_constants.chothia_list

    for i, position in enumerate(template_codes):
        position_dict[i] = expected_positions.index(position)
        positions_used.add(position)
    unused_positions = [i for i in range(len(expected_positions)) if
                        expected_positions[i] not in positions_used]
    os.chdir(start_dir)
    return position_dict, unused_positions

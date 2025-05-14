"""Converts saved weights for the final model into an xGPR model."""
import os
import numpy as np
from xGPR import xGPRegression, build_regression_dataset


def build_model_from_savez(project_dir, savez_fname, kernel,
        kernel_settings):
    """Loads the supplied savez file and sets up an xGPR model with the
    correct number of params. This is a messy process because xGPR has
    not yet been equipped with a save_state_dict method (this should
    be added in a future release), so that models are usually saved by
    pickling. Pickling however is not secure for the user who may
    be trying to reproduce the experiments in this repo so we are
    trying to avoid using it here (numpy savez only uses pickle if
    objects are supplied). We wanted the user to be able to
    download our model but had to find a way to do this that
    did not involve pickle hence this very convoluted workaround.
    We are very aware that the code here needs
    to be cleaned up / replaced with a cleaner method (TODO).

    Args:
        home_dir (str): A path to the project directory.
        savez_fname (str): The name of the savez file to load.
        kernel (str): The kernel choice. Must be the same as the
            original kernel choice.
        kernel_settings (dict): The kernel settings. Should match
            those originally used.

    Returns:
        model: An xGPRegression model.
    """
    fdict = np.load(savez_fname)
    current_dir = os.getcwd()

    # We need to supply a sample of what the data looks like when setting hyperparameters
    # for xGPR.
    sample_x_file = os.path.join(project_dir, "encoded_data", "encoded_seqs", "high",
            "train", "onehotESM_0_concat.npy")
    sample_y_file = os.path.join(project_dir, "encoded_data", "encoded_seqs", "high",
            "train", "enrich_0_y.npy")

    sample_regdata = build_regression_dataset([sample_x_file], [sample_y_file])

    xgp_model = xGPRegression(num_rffs = fdict['weights'].shape[0],
            variance_rffs = fdict['variance'].shape[0],
            kernel_choice = kernel, kernel_settings = kernel_settings,
            random_seed = fdict['random_seed'][0])
    xgp_model.set_hyperparams(fdict['hyperparams'], sample_regdata)

    # As long as random seed is the same, the kernel radem diag etc. will be unchanged.
    # We just need to make sure the model weights, variance and ymean, ystd are set up
    # correctly.
    xgp_model.weights = fdict['weights']
    xgp_model.var = fdict['variance']
    xgp_model.trainy_mean = fdict['trainy_mean'][0]
    xgp_model.trainy_std = fdict['trainy_std'][0]

    xgp_model.device = "gpu"
    os.chdir(current_dir)
    return xgp_model

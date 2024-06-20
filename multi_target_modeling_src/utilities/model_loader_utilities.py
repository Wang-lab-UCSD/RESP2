"""Contains resources for loading saved models."""
import os
import pickle
import torch
from xGPR import FastConv1d
from ..model_code.task_adapted_autoencoder import TaskAdaptedAutoencoder as TAA


def load_autoencoder(start_dir):
    """Loads the autoencoder from Parkinson et al 2023."""
    os.chdir(os.path.join(start_dir, "results_and_resources", "trained_models"))
    autoencoder = TAA()
    autoencoder.load_state_dict(torch.load("TaskAdapted_Autoencoder.ptc"))
    autoencoder.eval()
    os.chdir(start_dir)
    return autoencoder



def load_final_models(start_dir):
    """Loads the final models for simulated annealing."""
    os.chdir(os.path.join(start_dir, "results_and_resources", "trained_models"))
    if "FINAL_high_model.pk" not in os.listdir() or "FINAL_super_model.pk" \
            not in os.listdir():
        raise ValueError("Final models have not yet been constructed or "
                "have been removed.")

    with open("FINAL_high_model.pk", "rb") as fhandle:
        high_model = pickle.load(fhandle)["model"]
    with open("FINAL_super_model.pk", "rb") as fhandle:
        super_model = pickle.load(fhandle)["model"]

    os.chdir(start_dir)
    return high_model, super_model



def get_fc_encoder(random_seed = 123, seq_width = 21):
    """Returns an xGPR static layer for encoding data with pre-set
    parameters."""
    fc_encoder = FastConv1d(seq_width = seq_width,
                             device = "gpu", conv_width = 13,
                             num_features = 3000,
                             random_seed = random_seed)
    return fc_encoder

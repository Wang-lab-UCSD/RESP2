"""Contains resources for loading saved models."""
import os
import pickle
import torch
import wget
from xGPR import FastConv1d
from .load_savez_weights import build_model_from_savez
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
        if "FINAL_high_model.npz" not in os.listdir() or "FINAL_super_model.npz" \
                not in os.listdir():
            raw_high = wget.download("https://www.dropbox.com/scl/fi/4v2aoy1n1gm9enphij67u/FINAL_high_model.npz?rlkey=7s72ydreqwwwa4dtzgut83e4w&st=9ffo4hwx&dl=1")
            high_model = build_model_from_savez(start_dir, raw_high, "MiniARD",
                    {"intercept":True, "split_points":[104*21]})
            with open("FINAL_high_model.pk", "wb") as fhandle:
                pickle.dump(high_model, fhandle)
            os.remove(raw_high)

            raw_super = wget.download("https://www.dropbox.com/scl/fi/kle84kszmb8gksfict7o4/FINAL_super_model.npz?rlkey=2luvl9vskahp16h42ico7hi4z&st=lgq1jg38&dl=1")
            super_model = build_model_from_savez(start_dir, raw_super, "MiniARD",
                    {"intercept":True, "split_points":[104*21]})
            with open("FINAL_super_model.pk", "wb") as fhandle:
                pickle.dump(super_model, fhandle)
            os.remove(raw_super)

    with open("FINAL_high_model.pk", "rb") as fhandle:
        high_model = pickle.load(fhandle)
    with open("FINAL_super_model.pk", "rb") as fhandle:
        super_model = pickle.load(fhandle)

    os.chdir(start_dir)
    return high_model, super_model



def get_fc_encoder(random_seed = 123, seq_width = 21):
    """Returns an xGPR static layer for encoding data with pre-set
    parameters."""
    # Note that in xGPR v0.4.5 conv_width is an int not a list.
    fc_encoder = FastConv1d(seq_width = seq_width,
                             device = "gpu", conv_width = [13],
                             num_features = 3000,
                             random_seed = random_seed)
    return fc_encoder

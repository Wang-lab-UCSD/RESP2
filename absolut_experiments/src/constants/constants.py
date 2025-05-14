"""Contains useful constants for the resp in silico search."""

AAS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
       "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]


TARGET_PROTEIN_GROUPS = {"neuraminidase":["1NCA_N", "4QNP_A"],
        "il2":["4YUE_C", "5LQB_A"],
        "prion":["1TQB_A", "2W9E_A", "4H88_A"],
        "dust_mite":["3RVV_A", "4PP1_A"],
        "notch":["3L95_X", "5CZV_A"],
        "tissue_factor":["1JPS_T", "4M7L_T"]
}


CNN_MODEL_PARAMS = {"input_dim":21,
                    "hidden_dim":64,
                    "n_layers":2,
                    "kernel_size":11,
                    "dil_factor":1,
                    "rep_dim":100,
                    "pool_type":"max",
                    "weight_decay":0.,
                    "lr":0.001,
                    "n_epochs":100,
                    "gp_ridge_penalty":0.1,
                    "gp_amplitude":2}


XGPR_MODEL_PARAMS = {"5LQB_A":[-1.5383, -8.0645],
                     "4YUE_C":[-2.3747, -8.4336],
                     "1NCA_N":[-1.5511, -8.0589],
                     "4QNP_A":[-1.6105, -8.0954],
                     "1TQB_A":[-2.4823, -8.3473],
                     "2W9E_A":[-1.4268, -8.0146],
                     "4H88_A":[-1.853,  -8.1469],
                     "3RVV_A":[-2.6241, -8.4533],
                     "4PP1_A":[-1.2525, -7.9667],
                     "3L95_X":[-1.5642, -8.0351],
                     "5CZV_A":[-2.4485, -8.4471],
                     "1JPS_T":[-2.6852, -8.4126],
                     "4M7L_T":[-1.5745, -8.0014],
                     }



# The longest sequence length in any sub-dataset. Useful
# for the vbnn, which needs fixed-length input, so input
# length must be specified.
MAX_LENGTH = 43

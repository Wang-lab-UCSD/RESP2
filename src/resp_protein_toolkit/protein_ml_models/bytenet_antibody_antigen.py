'''A CNN architecture for regression and classification. Adapted
from Microsoft's CARP which is adapted from ByteNet. Can use a
last-layer Gaussian process to improve uncertainty calibration
and provide variance estimates for regression if so specified
by user. This variant is used when there is a pair of sequences
that vary -- for example a binder that is varied and a target
that is varied.'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm
from ..classic_rffs import VanillaRFFLayer
from .bytenet_antibody_only import PositionFeedForward, ByteNetBlock





class ByteNetPairedSeqs(torch.nn.Module):
    """A model for predicting the fitness of a given antibody-
    antigen pair using a series of ByteNet blocks. Note that it accepts
    two sets of sequences as input: the antigen sequence and the antibody
    sequence. Each of these is fed through its own series of ByteNet
    blocks, then at the end the representations of the two are
    merged.

    Args:
        input_dim (int): The expected dimensionality of the input, which is
            (N, L, input_dim).
        hidden_dim (int): The dimensions used inside the model.
        n_layers (int): The number of ByteNet blocks to use.
        kernel_size (int): The kernel width for ByteNet blocks.
        dil_factor (int): Used for calculating dilation factor, which increases by
            this factor on each subsequent layer. For short sequence inputs, use 1.
            For long sequences, 2 (or even 3) may be more appropriate.
        rep_dim (int): At the end of the ByteNet blocks, the mean is taken across
            the tokens in each sequence to generate a representation. rep_dim
            determines the size of that representation.
        dropout (float): The level of dropout to apply.
        slim (bool): If True, use a smaller size within each ByteNet block.
        llgp (bool): If True, use a last-layer GP.
        antigen_dim: Either None or an int. If None, the antigen input is assumed
            to have the same dimensionality as the antibody.
        objective (str): Must be one of "regression", "binary_classifier",
            "multiclass".
        num_predicted_categories (int): The number of categories (i.e. possible values
            for y in output). Ignored unless objective is "multiclass".
        gp_cov_momentum (float): A "discount factor" used to update a moving average
            for the updates to the covariance matrix when llgp is True. 0.999 is a
            reasonable default if the number of steps per epoch is large, otherwise
            you may want to experiment with smaller values. If you set this to < 0
            (e.g. to -1), the precision matrix will be generated in a single epoch
            without any momentum. If llgp is False (model is not uncertainty aware),
            there is no covariance matrix and this argument is ignored.
        gp_ridge_penalty (float): The ridge penalty for the last layer GP. Performance
            is not usually very sensitive to this although in some cases experimenting
            with it may improve performance, and it can affect calibration. It should
            not be set to zero since it is important for numerical stability for it to
            be > 0. The default is 1e-3.
        gp_amplitude (float): The kernel amplitude for the last layer Gaussian process.
            This is the inverse of the lengthscale. Performance is not generally
            very sensitive to the selected value for this hyperparameter,
            although it may affect calibration. Defaults to 1.
        num_rffs (int): The number of random Fourier features used to approximate a GP
            in the final model layer. Only used if llgp is set to True; otherwise it is
            ignored. A larger number of RFFs means a more accurate kernel approximation.
            Default is 1024 which is usually fine for most purposes.
    """
    def __init__(self, input_dim, hidden_dim, n_layers, kernel_size, dil_factor,
                rep_dim = 100, dropout = 0.0, slim = False, llgp = False,
                antigen_dim = None, objective = "regression",
                num_predicted_categories = 1, gp_cov_momentum = 0.999,
                gp_ridge_penalty=1e-3, gp_amplitude=1.,
                num_rffs = 1024):
        super().__init__()
        torch.manual_seed(123)
        torch.backends.cudnn.deterministic = True
        use_spectral_norm = llgp

        self.objective = objective
        if objective == "multiclass":
            nclasses = num_predicted_categories
            likelihood = "multiclass"
            if num_predicted_categories <= 2:
                raise RuntimeError("If running in multiclass mode, "
                        "num_predicted_categories must always be > 2. "
                        "If there are only two possible categories, "
                        "binary classification is more appropriate.")
        elif objective == "binary_classifier":
            nclasses = 1
            likelihood = "binary_logistic"
        elif objective == "regression":
            nclasses = 1
            likelihood = "Gaussian"
        else:
            raise RuntimeError("Unrecognized objective supplied.")

        if llgp:
            torch.cuda.manual_seed(123)
            torch.use_deterministic_algorithms(True)

        # Calculate the dilation factors for subsequent layers
        dil_log2 = int(np.log2(dil_factor)) + 1
        dilations = [2 ** (n % dil_log2) for n in range(n_layers)]
        d_h = hidden_dim
        if slim:
            d_h = d_h // 2

        self.adjuster = PositionFeedForward(input_dim, hidden_dim, use_spectral_norm)
        if antigen_dim is not None:
            self.antigen_adjuster = PositionFeedForward(antigen_dim, hidden_dim,
                                                        use_spectral_norm)
        else:
            self.antigen_adjuster = None

        antibody_layers = [
            ByteNetBlock(hidden_dim, d_h, hidden_dim, kernel_size, dilation=d,
                         use_spectral_norm = use_spectral_norm)
            for d in dilations
        ]
        self.antibody_layers = torch.nn.ModuleList(modules=antibody_layers)

        antigen_layers = [
            ByteNetBlock(hidden_dim, d_h, hidden_dim, kernel_size, dilation=d,
                         use_spectral_norm = use_spectral_norm)
            for d in dilations
        ]
        self.antigen_layers = torch.nn.ModuleList(modules=antigen_layers)

        self.down_adjuster = PositionFeedForward(hidden_dim, rep_dim,
                                            use_spectral_norm = use_spectral_norm)


        if llgp:
            self.out_layer = VanillaRFFLayer(in_features = 2 * rep_dim,
                        RFFs = num_rffs, out_targets = nclasses, gp_cov_momentum = gp_cov_momentum,
                        gp_ridge_penalty = gp_ridge_penalty, likelihood = likelihood,
                        random_seed = 123, amplitude = gp_amplitude)
        else:
            if use_spectral_norm:
                self.out_layer = SpectralNorm(torch.nn.Linear(2 * rep_dim, nclasses))
            else:
                self.out_layer = torch.nn.Linear(2 * rep_dim, nclasses)

        self.dropout = dropout
        self.llgp = llgp


    def forward(self, x_antibody, x_ant,
                update_precision = False, get_var = False):
        """
        Args:
            x_antibody (N, L, in_channels): -- the antibody sequence data
            x_ant (N, L2, in_channels): -- the antigen sequence data
            update_precision (bool): If you want to generate the covariance matrix
                during the last epoch only (i.e. you set gp_cov_momentum to < 0
                when creating this model), set this to True during the last epoch
                only. If you want to generate the covariance matrix over the course
                of training (i.e. gp_cov_momentum is > 0 and < 1), set this to True
                throughout training. This should always be False during inference.
            get_var (bool): If True, return estimated variance on predictions.
                Only available if 'llgp' in class constructor is True AND objective
                is regression. Otherwise, this option can still be passed but
                will be ignored.

        Returns:
            Output (tensor): -- Shape depends on objective. If regression or
                binary_classifier, shape will be (N). If multiclass, shape
                will be (N, num_predicted_classes) that was passed when the
                model was constructed.
            var (tensor): Only returned if get_var is True, objective is regression
                and model was initialized with llgp set to True. If returned, it
                is a tensor of shape (N).
        """
        x_antibody = self.adjuster(x_antibody)
        if self.antigen_adjuster is not None:
            x_antigen = self.antigen_adjuster(x_ant)
        else:
            x_antigen = self.adjuster(x_ant)

        for layer in self.antibody_layers:
            x_antibody = layer(x_antibody)
            if self.dropout > 0.0 and self.training:
                x_antibody = F.dropout(x_antibody, self.dropout)
        for layer in self.antigen_layers:
            x_antigen = layer(x_antigen)
            if self.dropout > 0.0 and self.training:
                x_antigen = F.dropout(x_antigen, self.dropout)

        x_antibody = self.down_adjuster(x_antibody)
        x_antigen = self.down_adjuster(x_antigen)
        xdata = torch.cat([x_antibody, x_antigen], dim=2)
        xdata = torch.mean(xdata, dim=1)

        if self.objective == "regression":
            if self.llgp:
                if get_var:
                    preds, var = self.out_layer(x_antibody, get_var = get_var)
                    return preds.squeeze(1), var
                preds = self.out_layer(x_antibody, update_precision)
            else:
                preds = self.out_layer(x_antibody)
            return preds.squeeze(1)
        if self.objective == "binary_classifier":
            if self.llgp and get_var:
                preds, var = self.out_layer(x_antibody, get_var = get_var)
                return F.sigmoid(preds.squeeze(1)), var
            if self.llgp:
                preds = self.out_layer(x_antibody, update_precision)
            else:
                preds = self.out_layer(x_antibody)
            return F.sigmoid(preds.squeeze(1))
        if self.objective == "multiclass":
            if self.llgp and get_var:
                preds, var = self.out_layer(x_antibody, get_var = get_var)
                return F.softmax(preds, dim=1), var
            if self.llgp:
                preds = self.out_layer(x_antibody, update_precision)
            else:
                preds = self.out_layer(x_antibody)
            return F.softmax(preds, dim=1)

        # Double-check that the objective is correct to avoid weird
        # errors...
        raise RuntimeError("Model was initialized with an invalid task / objective.")





    def predict(self, x, ant, get_var = False):
        """This function returns the predicted y-value for each
        datapoint. For convenience, it takes numpy arrays as input
        and returns numpy arrays as output. If you already have
        PyTorch tensors it may be slightly faster / more convenient
        to use forward instead of calling predict.

        Args:
            x (np.ndarray): The input antibody data.
            ant (np.ndarray): the input antigen data.
            get_var (bool): If True, return estimated variance on predictions.
                Only available if 'llgp' in class constructor is True and the
                objective in the class constructor is "regression". Otherwise
                this argument is ignored.

        Returns:
            scores (np.ndarray): If class objective is "regression" or
                "binary_classifier", this is of shape (N). If "multiclass",
                this is of shape (N, num_predicted_classes) from the
                class constructor.
            var (np.ndarray): Only returned if get_var is True, llgp in
                the class constructor is True and the objective is "regression".
                If returned, is of shape (N).
        """
        with torch.no_grad():
            self.eval()
            x = torch.from_numpy(x).float()
            ant = torch.from_numpy(ant).float()
            if next(self.parameters()).is_cuda:
                x = x.cuda()
                ant = ant.cuda()
            if self.llgp and get_var:
                preds, var = self.forward(x, ant, get_var = get_var)
                return preds.cpu().numpy(), var.cpu().numpy()
            return self.forward(x, ant).cpu().numpy()

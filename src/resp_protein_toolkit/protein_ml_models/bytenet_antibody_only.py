'''A CNN architecture for regression and classification. Adapted
from Microsoft's CARP which is adapted from ByteNet. Can use a
last-layer Gaussian process to improve uncertainty calibration
and provide variance estimates for regression if so specified
by user. This architecture is used when there is only one
sequence that is varied (e.g. an antibody that is mutated
while the target remains the same).'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm
from ..classic_rffs import VanillaRFFLayer


class ConvLayer(torch.nn.Conv1d):
    """ A 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, but applies automatic padding
    for convenience, and automatically performs transposition on inputs.

    
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """Class constructor.

        Args:
            param in_channels: input channels
            param out_channels: output channels
            param kernel_size: the kernel width
            param stride: filter shift
            param dilation: dilation factor
            param groups: perform depth-wise convolutions
            param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, dilation=dilation,
                                            groups=groups, bias=bias, padding=padding)

    def forward(self, xdata):
        """Standard forward pass.
        Args:
            Input: (N, L, in_channels)
        Returns:
            Output: (N, L, out_channels)
        """
        return super().forward(xdata.transpose(1, 2)).transpose(1, 2)



class PositionFeedForward(torch.nn.Module):
    """A feed-forward layer for the bytenet block.

    Args:
        d_in: The input dimensionality.
        d_out: The output dimensionality.
        use_spectral_norm (bool): If True, use spectral norm on the weights.
    """

    def __init__(self, d_in, d_out, use_spectral_norm = False):
        super().__init__()
        if use_spectral_norm:
            self.conv = SpectralNorm(torch.nn.Conv1d(d_in, d_out, 1))
        else:
            self.conv = torch.nn.Conv1d(d_in, d_out, 1)
        self.factorized = False


    def forward(self, xdata):
        """The forward pass.

        Args:
            Input: (N, L, in_channels)
        Returns:
            Output: (N, L, out_channels)
        """
        return self.conv(xdata.transpose(1, 2)).transpose(1, 2)



class ByteNetBlock(torch.nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).

    Args:
        d_in (int): the input dimensionality
        d_h (int): The within-block hidden dimensionality
        d_out (int): The output dimensionality
        kernel_size (int): the size of the convolution kernel
        dilation (int): The convolution kernel dilation
        groups (int): depth-wise convolutions (if desired)
        use_spectral_norm (bool): If True, use spectral norm on the weights.
    """

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1,
                 use_spectral_norm = False):
        super().__init__()
        if use_spectral_norm:
            self.conv = SpectralNorm(ConvLayer(d_h, d_h, kernel_size=kernel_size,
                                           dilation=dilation, groups=groups))
        else:
            self.conv = ConvLayer(d_h, d_h, kernel_size=kernel_size,
                                           dilation=dilation, groups=groups)

        layers1 = [
            torch.nn.LayerNorm(d_in),
            torch.nn.GELU(),
            PositionFeedForward(d_in, d_h, use_spectral_norm),
            torch.nn.LayerNorm(d_h),
            torch.nn.GELU()
        ]
        layers2 = [
            torch.nn.LayerNorm(d_h),
            torch.nn.GELU(),
            PositionFeedForward(d_h, d_out, use_spectral_norm),
        ]
        self.sequence1 = torch.nn.Sequential(*layers1)
        self.sequence2 = torch.nn.Sequential(*layers2)


    def forward(self, xdata):
        """
        Args:
            Input: (N, L, in_channels)
        Returns:
            Output: (N, L, out_channels)
        """
        return xdata + self.sequence2(
            self.conv(self.sequence1(xdata)))


class ByteNetSingleSeq(torch.nn.Module):
    """A model for predicting the fitness of a given antibody
    using a series of ByteNet blocks. Note that it makes predictions
    using a single sequence only, not sequence pairs.

    Args:
        input_dim (int): The expected dimensionality of the input, which is
            (N, L, input_dim).
        hidden_dim (int): The dimensions used inside the model.
        n_layers (int): The number of ByteNet blocks to use.
        kernel_size (int): The kernel width for ByteNet blocks.
        dil_factor (int): Used for calculating dilation factor, which increases by
            this factor on each subsequent layer. For short sequence inputs, use 1.
            For long sequences, 2 (or even 3) may be more appropriate.
        rep_dim (int): At the end of the ByteNet blocks, the model either average
            pools or maxpools across the tokens in each sequence to generate a representation.
            rep_dim determines the size of that representation.
        pool_type (str): One of "max", "mean". Determines the type of pooling
            that is applied in the final layer.
        dropout (float): The level of dropout to apply.
        slim (bool): If True, use a smaller size within each ByteNet block.
        llgp (bool): If True, use a last-layer GP, which enables us to estimate
            uncertainty.
        objective (str): Must be one of "regression", "binary_classifier",
            "multiclass", "ordinal".
        num_predicted_categories (int): The number of categories (i.e. possible values
            for y in output). Ignored unless objective is "multiclass" or "ordinal",
            and required if the objective is either of those things.
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
                rep_dim = 100, pool_type = "max", dropout = 0.0, slim = False,
                llgp = False, objective = "regression", num_predicted_categories = 1,
                gp_cov_momentum = 0.999, gp_ridge_penalty=1e-3, gp_amplitude=1.,
                num_rffs = 1024):
        super().__init__()
        torch.manual_seed(123)
        torch.backends.cudnn.deterministic = True
        use_spectral_norm = llgp

        self.rep_dim = rep_dim
        self.pool_type = pool_type
        if pool_type not in ("max", "mean"):
            raise RuntimeError("Pool type must be one of 'max', 'mean'.")

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
            likelihood = "gaussian"
        elif objective == "ordinal":
            nthresholds = num_predicted_categories - 1
            nclasses = 1
            if nthresholds < 1:
                raise RuntimeError("The number of categories must be >= 2 "
                        "to use ordinal regression.")

            likelihood = "gaussian"
            class_thresholds = torch.arange(nthresholds).float()
            class_thresholds -= torch.mean(class_thresholds)
            self.register_buffer("class_thresholds", class_thresholds)
        else:
            raise RuntimeError("Unrecognized objective supplied.")

        if llgp:
            torch.cuda.manual_seed(123)

        # Calculate the dilation factors for subsequent layers
        dil_log2 = int(np.log2(dil_factor)) + 1
        dilations = [2 ** (n % dil_log2) for n in range(n_layers)]
        d_h = hidden_dim
        if slim:
            d_h = d_h // 2

        self.adjuster = PositionFeedForward(input_dim, hidden_dim, use_spectral_norm)

        antibody_layers = [
            ByteNetBlock(hidden_dim, d_h, hidden_dim, kernel_size, dilation=d,
                         use_spectral_norm = use_spectral_norm)
            for d in dilations
        ]
        self.antibody_layers = torch.nn.ModuleList(modules=antibody_layers)

        self.down_adjuster = PositionFeedForward(hidden_dim, rep_dim,
                                            use_spectral_norm = use_spectral_norm)

        self.out_norm = torch.nn.BatchNorm1d(rep_dim)

        if llgp:
            self.out_layer = VanillaRFFLayer(in_features = rep_dim,
                        RFFs = num_rffs, out_targets = nclasses,
                        gp_cov_momentum = gp_cov_momentum,
                        gp_ridge_penalty = gp_ridge_penalty,
                        likelihood = likelihood,
                        random_seed = 123,
                        amplitude = gp_amplitude)
        else:
            if use_spectral_norm:
                self.out_layer = SpectralNorm(torch.nn.Linear(rep_dim, nclasses))
            else:
                self.out_layer = torch.nn.Linear(rep_dim, nclasses)

        self.dropout = dropout
        self.llgp = llgp




    def forward(self, x_antibody, update_precision = False, get_var = False):
        """
        Args:
            x_antibody (N, L, in_channels): -- the antibody sequence data
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

        for layer in self.antibody_layers:
            x_antibody = layer(x_antibody)
            if self.dropout > 0.0 and self.training:
                x_antibody = F.dropout(x_antibody, self.dropout)

        x_antibody = F.relu(self.down_adjuster(x_antibody))
        if self.pool_type == "max":
            x_antibody = torch.max(x_antibody, dim=1)[0]
        else:
            x_antibody = torch.mean(x_antibody, dim=1)

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
        if self.objective == "ordinal":
            if self.llgp and get_var:
                preds, var = self.out_layer(x_antibody, get_var = get_var)
                if len(preds.shape) == 1:
                    preds = preds.unsqueeze(1)
                preds = preds[:,0:1] - self.class_thresholds[None,:]
                return F.sigmoid(preds), var
            if self.llgp:
                preds = self.out_layer(x_antibody, update_precision)
            else:
                preds = self.out_layer(x_antibody)

            if len(preds.shape) == 1:
                preds = preds.unsqueeze(1)
            preds = preds[:,0:1] - self.class_thresholds[None,:]
            return F.sigmoid(preds)

        # Double-check that the objective is correct to avoid weird
        # errors...
        raise RuntimeError("Model was initialized with an invalid task / objective.")





    def predict(self, x, get_var = False):
        """This function returns the predicted y-value for each
        datapoint. For convenience, it takes numpy arrays as input
        and returns numpy arrays as output. If you already have
        PyTorch tensors it may be slightly faster / more convenient
        to use forward instead of calling predict.

        Args:
            x (np.ndarray): The input antibody data.
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
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            if self.llgp and get_var:
                preds, var = self.forward(x, get_var = get_var)
                return preds.cpu().numpy(), var.cpu().numpy()
            return self.forward(x).cpu().numpy()

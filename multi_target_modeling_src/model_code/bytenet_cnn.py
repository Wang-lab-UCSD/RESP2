'''A CNN architecture for regression. Adapted from Microsoft's CARP which
is adapted from ByteNet.'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm
from uncertaintyAwareDeepLearn import VanillaRFFLayer


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


class CNNRegModel(torch.nn.Module):
    """The overall model for predicting the fitness of a given mutant
    using a series of ByteNet blocks. Note that it accepts two sets
    of sequences as input: the antigen sequence and the antibody
    sequence. Each of these is fed through its own series of ByteNet
    blocks, then at the end the representations of the two are
    merged.

    Args:
        input_dim (int): The expected dimensionality of the input, which is
            (N, L, hidden_dim).
        hidden_dim (int): The dimensions used inside the model.
        n_layers (int): The number of ByteNet blocks to use.
        kernel_size (int): The kernel width for ByteNet blocks.
        dil_factor (int): Used for calculating dilation factor, which increases on
            subsequent layers.
        dropout (float): The level of dropout to apply.
        slim (bool): If True, use a smaller size within each ByteNet block.
        llgp (bool): If True, use a last-layer GP.
        antigen_dim: Either None or an int. If None, the antigen input is assumed
            to have the same dimensionality as the antibody.
        use_spectral_norm (bool): If True, use spectral norm on the weights.
        contextual_regression (bool): If True, use contextual regression. This
            cannot be set to True if SNGP is also set to True.
    """
    def __init__(self, input_dim, hidden_dim, n_layers, kernel_size, dil_factor,
                num_antibody_tokens, num_antigen_tokens, dropout = 0.0, slim = False,
                 llgp = False, antigen_dim = None, use_spectral_norm = False,
                 contextual_regression = False):
        super().__init__()
        torch.manual_seed(123)
        torch.backends.cudnn.deterministic = True

        if contextual_regression and llgp:
            raise RuntimeError("Contextual regression and llgp are mutually exclusive.")

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

        if not contextual_regression:
            self.down_adjuster = PositionFeedForward(hidden_dim, 1,
                                            use_spectral_norm = use_spectral_norm)
            self.final_lnorm = torch.nn.LayerNorm(num_antibody_tokens + num_antigen_tokens)
        else:
            self.down_adjuster = PositionFeedForward(hidden_dim, 21,
                                            use_spectral_norm = use_spectral_norm)


        if llgp:
            self.out_layer = VanillaRFFLayer(in_features = num_antibody_tokens + num_antigen_tokens,
                        RFFs = 1024, out_targets = 1, gp_cov_momentum = 0.999,
                        gp_ridge_penalty = 1e-3, likelihood = "gaussian",
                        random_seed = 123)
        else:
            if use_spectral_norm:
                self.out_layer = SpectralNorm(torch.nn.Linear(num_antibody_tokens +
                                                        num_antigen_tokens, 1))
            else:
                self.out_layer = torch.nn.Linear(num_antibody_tokens +
                                                        num_antigen_tokens, 1)

        self.dropout = dropout
        self.llgp = llgp
        self.contextual_regression = contextual_regression


    def forward(self, x_antibody, x_ant,
                update_precision = False, get_var = False):
        """
        Args:
            x_antibody: (N, L, in_channels) -- the antibody sequence data
            x_ant: (N, L2, in_channels) -- the antigen sequence data
            update_precision (bool): Should be True during training, False
                otherwise.
            get_var (bool): If True, return estimated variance on predictions.
                Only available if 'llgp' in class constructor is True.

        Returns:
            Output: (N)
        """
        if self.contextual_regression:
            x_init = torch.cat([x_antibody.clone(), x_ant.clone()], dim=1)

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

        if self.contextual_regression:
            xdata = torch.cat([x_antibody, x_antigen], dim=1)
            return (xdata * x_init).sum(dim=2).sum(dim=1)

        xdata = self.final_lnorm(torch.cat([x_antibody, x_antigen], dim=1).squeeze(2))
        if self.llgp:
            if get_var:
                preds, var = self.out_layer(xdata, get_var = get_var)
                return preds.squeeze(1), var
            preds = self.out_layer(xdata, update_precision)
        else:
            preds = self.out_layer(xdata)
        return preds.squeeze(1)


    def extract_representation(self, x_antibody, x_ant):
        """Extracts a representation of the input. Only available if
        contextual regression was selected.

        Args:
            x_antibody: (N, L, in_channels) -- the antibody sequence data
            x_ant: (N, L2, in_channels) -- the antigen sequence data

        Returns:
            Output: Same size as x_antibody and x_ant concat along dim1.
        """
        if not self.contextual_regression:
            raise RuntimeError("This option is only available if using "
                    "contextual regression.")
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

        return torch.cat([x_antibody, x_antigen], dim=1)



    def predict(self, x, ant, get_var = False):
        """This function returns the predicted y-value for each
        datapoint.

        Args:
            x (tensor): The input antibody data.
            ant (tensor): the input antigen data.
            get_var (bool): If True, return estimated variance on predictions.
                Only available if 'llgp' in class constructor is True.

        Returns:
            scores (tensor): predicted y-values
        """
        with torch.no_grad():
            self.eval()
            x = x.float().cuda()
            ant = ant.float().cuda()
            if self.llgp and get_var:
                preds, var = self.forward(x, ant, get_var = get_var)
                return preds.cpu(), var.cpu()
            return self.forward(x, ant).cpu()
